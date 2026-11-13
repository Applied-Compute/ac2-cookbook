from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from dataclasses import dataclass
from typing import Annotated, cast

import modal
from ac2.harness.types import (
    GradingStatus,
    GradingStatusResponse,
    HarnessTask,
    RolloutRequest,
    RolloutStatus,
    ScoreResponse,
    StatusResponse,
    SubmitGradingRequest,
    SubmitPayload,
    SubmitResponse,
)
from ac2.runtime import Message
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, JsonValue, ValidationError
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

WORKDIR = "/testbed"
OPENCODE_VERSION = "1.15.11"
OPENCODE = "/opt/opencode/opencode"
SANDBOX_TIMEOUT_SECONDS = 2 * 60 * 60
ROLLOUT_TIMEOUT_SECONDS = 90 * 60
GRADER_TIMEOUT_SECONDS = 1_500
MODAL_TIMEOUT_RETURNCODE = -1
OUTPUT_TAIL_CHARS = 4_000


class SwebenchTaskParams(BaseModel):
    datapoint: dict[str, JsonValue]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Rollout:
    sandbox_id: str
    image_head: str
    patch: str
    opencode: CommandResult
    error: str | None


app = FastAPI()
rollouts: dict[str, asyncio.Task[Rollout]] = {}
rollout_ids_by_key: dict[str, str] = {}
gradings: dict[str, asyncio.Task[ScoreResponse]] = {}


@app.post("/submit", response_model=SubmitResponse)
async def submit(
    request: SubmitPayload,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> SubmitResponse:
    if idempotency_key not in rollout_ids_by_key:
        datapoint = _datapoint(request.task.env_params)
        prompt = _single_user_prompt(request.task)
        rollout_id = f"swebench_{uuid.uuid4().hex}"
        rollouts[rollout_id] = asyncio.create_task(
            run_rollout(request, datapoint, prompt)
        )
        rollout_ids_by_key[idempotency_key] = rollout_id
    return SubmitResponse(rollout_id=rollout_ids_by_key[idempotency_key])


@app.post("/get_status", response_model=StatusResponse)
async def get_status(request: RolloutRequest) -> StatusResponse:
    rollout = _rollout(request.rollout_id)
    if not rollout.done():
        return StatusResponse(status=RolloutStatus.running)
    if rollout.cancelled():
        return StatusResponse(
            status=RolloutStatus.failed, error_message="OpenCode was cancelled"
        )
    if error := rollout.exception():
        return StatusResponse(
            status=RolloutStatus.failed, error_message=str(error)[:2000]
        )
    if error := rollout.result().error:
        return StatusResponse(status=RolloutStatus.failed, error_message=error[:2000])
    return StatusResponse(status=RolloutStatus.completed)


@app.post("/collect_outputs")
async def collect_outputs(request: RolloutRequest) -> dict[str, JsonValue] | None:
    task = _rollout(request.rollout_id)
    await asyncio.wait([task])
    if task.cancelled() or task.exception():
        return None
    rollout = task.result()
    return {
        "sandbox_id": rollout.sandbox_id,
        "model_patch_bytes": len(rollout.patch.encode()),
        "returncode": rollout.opencode.returncode,
        "stdout": rollout.opencode.stdout,
        "stderr": rollout.opencode.stderr,
    }


@app.post("/submit_grading")
async def submit_grading(request: SubmitGradingRequest) -> dict[str, JsonValue]:
    if request.rollout_id not in gradings:
        rollout = _rollout(request.rollout_id)
        datapoint = _datapoint(request.task.env_params)
        gradings[request.rollout_id] = asyncio.create_task(grade(rollout, datapoint))
    return {}


@app.post("/get_grading_status", response_model=GradingStatusResponse)
async def get_grading_status(request: RolloutRequest) -> GradingStatusResponse:
    grading = _grading(request.rollout_id)
    if not grading.done():
        return GradingStatusResponse(status=GradingStatus.running)
    if grading.cancelled():
        return GradingStatusResponse(
            status=GradingStatus.failed, error_message="Grading was cancelled"
        )
    if error := grading.exception():
        return GradingStatusResponse(
            status=GradingStatus.failed, error_message=str(error)[:2000]
        )
    return GradingStatusResponse(status=GradingStatus.completed)


@app.post("/collect_score", response_model=ScoreResponse)
async def collect_score(request: RolloutRequest) -> ScoreResponse:
    return await _grading(request.rollout_id)


async def run_rollout(
    request: SubmitPayload, datapoint: dict[str, JsonValue], prompt: str
) -> Rollout:
    test_spec = _test_spec(datapoint)
    instruction = (
        "Solve this SWE-bench issue in /testbed. Make the smallest correct change, "
        "do not modify tests, and run the relevant tests.\n\n"
        f"{datapoint['problem_statement']}\n"
    )
    opencode_url = (
        f"https://github.com/anomalyco/opencode/releases/download/v{OPENCODE_VERSION}/"
        "opencode-linux-x64-baseline.tar.gz"
    )
    image = (
        modal.Image.from_registry(test_spec.instance_image_key)
        .entrypoint([])
        .run_commands(
            "mkdir -p /opt/opencode",
            f"curl --fail --silent --show-error --location {opencode_url} "
            "--output /tmp/opencode.tar.gz",
            "tar -xzf /tmp/opencode.tar.gz -C /opt/opencode",
            "chmod +x /opt/opencode/opencode",
        )
    )
    sandbox = await create_sandbox(image)
    try:
        head = await run_command(sandbox, "git rev-parse HEAD")
        if head.returncode != 0:
            raise RuntimeError(f"Failed to read image commit: {head.stderr}")

        setup = await run_command(sandbox, "mkdir -p /task")
        if setup.returncode != 0:
            raise RuntimeError(f"Failed to prepare harness sandbox: {setup.stderr}")
        await sandbox.filesystem.write_text.aio(instruction, "/task/instruction.md")

        config = {
            "$schema": "https://opencode.ai/config.json",
            "agent": {"title": {"disable": True}},
            "provider": {
                "ac2": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "AC2",
                    "options": {
                        "apiKey": "{env:AC2_ROLLOUT_API_KEY}",
                        "baseURL": request.ac_api_url,
                    },
                    "models": {"model": {"name": "Model selected by AC2"}},
                }
            },
        }
        opencode = await run_command(
            sandbox,
            (
                f"{OPENCODE} run --model ac2/model --format json "
                '--dangerously-skip-permissions "$OPENCODE_PROMPT" </dev/null'
            ),
            env={
                "AC2_ROLLOUT_API_KEY": request.ac_api_key,
                "OPENCODE_PROMPT": prompt,
                "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
            },
            timeout=ROLLOUT_TIMEOUT_SECONDS,
        )
        patch_result = await run_command(
            sandbox,
            """
set -euo pipefail
git add --intent-to-add .
git diff --binary "$IMAGE_HEAD"
""",
            env={"IMAGE_HEAD": head.stdout.strip()},
        )
        if patch_result.returncode != 0:
            raise RuntimeError(f"Failed to fetch model patch: {patch_result.stderr}")
    finally:
        await sandbox.terminate.aio()

    error = _opencode_error(opencode)
    if error and not patch_result.stdout:
        raise RuntimeError(error)
    return Rollout(
        sandbox_id=sandbox.object_id,
        image_head=head.stdout.strip(),
        patch=patch_result.stdout,
        opencode=CommandResult(
            returncode=opencode.returncode,
            stdout=opencode.stdout[-OUTPUT_TAIL_CHARS:],
            stderr=opencode.stderr[-OUTPUT_TAIL_CHARS:],
        ),
        error=error,
    )


async def grade(
    rollout_task: asyncio.Task[Rollout], datapoint: dict[str, JsonValue]
) -> ScoreResponse:
    rollout = await rollout_task
    test_spec = _test_spec(datapoint)
    sandbox = await create_sandbox(
        modal.Image.from_registry(test_spec.instance_image_key).entrypoint([])
    )
    try:
        base = await run_command(
            sandbox,
            "git rev-parse HEAD && git status --porcelain --untracked-files=no",
        )
        if base.returncode != 0 or base.stdout.strip() != rollout.image_head:
            raise RuntimeError(
                f"Grading sandbox is not a clean checkout of {rollout.image_head}: "
                f"{(base.stdout + base.stderr).strip()}"
            )

        if rollout.patch:
            await sandbox.filesystem.write_text.aio(rollout.patch, "/tmp/model.patch")
            check = await run_command(
                sandbox, "git apply --check --binary /tmp/model.patch"
            )
            if check.returncode != 0:
                return ScoreResponse(
                    score=0.0,
                    reasoning=(
                        "model patch does not apply to the task's base commit; "
                        "scored as an invalid submission"
                    ),
                    artifacts={"resolved": False, "patch_error": check.stderr[:2_000]},
                )
            apply = await run_command(sandbox, "git apply --binary /tmp/model.patch")
            if apply.returncode != 0:
                raise RuntimeError(f"Failed to apply model patch: {apply.stderr}")

        await sandbox.filesystem.write_text.aio(test_spec.eval_script, "/root/eval.sh")
        result = await run_command(
            sandbox,
            "cd /testbed && bash /root/eval.sh",
            timeout=GRADER_TIMEOUT_SECONDS,
        )
    finally:
        await sandbox.terminate.aio()

    if result.returncode == MODAL_TIMEOUT_RETURNCODE:
        return ScoreResponse(
            score=0.0,
            reasoning=(
                f"tests timed out after {GRADER_TIMEOUT_SECONDS} seconds; "
                "scored as unresolved"
            ),
            artifacts={"resolved": False, "tests_timed_out": True},
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as test_log:
        test_log.write(result.stdout)
        if result.stderr:
            test_log.write(f"\n{result.stderr}")
        test_log.flush()
        report = get_eval_report(
            test_spec=test_spec,
            prediction={
                "instance_id": test_spec.instance_id,
                "model_patch": "",
            },
            test_log_path=test_log.name,
            include_tests_status=True,
        )

    instance_report = report.get(test_spec.instance_id, {})
    if "tests_status" not in instance_report:
        raise RuntimeError(
            f"SWE-bench test output could not be parsed: {result.stdout[-1_000:]}"
        )
    resolved = bool(instance_report["resolved"])
    tests_status = instance_report["tests_status"]
    fail_to_pass = tests_status["FAIL_TO_PASS"]
    pass_to_pass = tests_status["PASS_TO_PASS"]
    return ScoreResponse(
        score=1.0 if resolved else 0.0,
        reasoning=(
            f"resolved={resolved}; "
            f"fail-to-pass={len(fail_to_pass['success'])}/"
            f"{len(fail_to_pass['success']) + len(fail_to_pass['failure'])}; "
            f"pass-to-pass={len(pass_to_pass['success'])}/"
            f"{len(pass_to_pass['success']) + len(pass_to_pass['failure'])}"
        ),
        artifacts={"resolved": resolved, "tests_status": tests_status},
    )


async def create_sandbox(image: modal.Image) -> modal.Sandbox:
    modal_app = await modal.App.lookup.aio("swebench-opencode", create_if_missing=True)
    return await modal.Sandbox.create.aio(
        "sleep",
        "infinity",
        app=modal_app,
        image=image,
        timeout=SANDBOX_TIMEOUT_SECONDS,
        workdir=WORKDIR,
        cpu=4,
        memory=16_384,
    )


async def run_command(
    sandbox: modal.Sandbox,
    command: str,
    *,
    env: dict[str, str | None] | None = None,
    timeout: int | None = None,
) -> CommandResult:
    proc = await sandbox.exec.aio(
        "bash",
        "-c",
        command,
        env=env,
        timeout=timeout,
        text=False,
    )
    returncode = await proc.wait.aio()
    stdout = (await proc.stdout.read.aio()).decode("utf-8", errors="replace")
    stderr = (await proc.stderr.read.aio()).decode("utf-8", errors="replace")
    return CommandResult(returncode=returncode, stdout=stdout, stderr=stderr)


def _opencode_error(opencode: CommandResult) -> str | None:
    if opencode.returncode != 0:
        return f"OpenCode exited with code {opencode.returncode}"
    lines = opencode.stdout.strip().splitlines()
    try:
        event = json.loads(lines[-1]) if lines else None
    except ValueError:
        return None
    if isinstance(event, dict) and event.get("type") == "error":
        return f"OpenCode reported an error: {json.dumps(event.get('error'))}"
    return None


def _rollout(rollout_id: str) -> asyncio.Task[Rollout]:
    rollout = rollouts.get(rollout_id)
    if rollout is None:
        raise HTTPException(status_code=404, detail="Unknown rollout")
    return rollout


def _grading(rollout_id: str) -> asyncio.Task[ScoreResponse]:
    grading = gradings.get(rollout_id)
    if grading is None:
        raise HTTPException(status_code=404, detail="Unknown grading")
    return grading


def _datapoint(env_params: dict[str, JsonValue] | None) -> dict[str, JsonValue]:
    try:
        return SwebenchTaskParams.model_validate(env_params or {}).datapoint
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail="Expected a SWE-bench datapoint"
        ) from exc


def _test_spec(datapoint: dict[str, JsonValue]) -> TestSpec:
    return make_test_spec(cast(SWEbenchInstance, datapoint), namespace="swebench")


def _single_user_prompt(task: HarnessTask) -> str:
    try:
        messages = [Message.model_validate(item) for item in task.input]
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail="OpenCode expects one text user message"
        ) from exc
    if (
        len(messages) != 1
        or messages[0].role != "user"
        or not isinstance(messages[0].content, str)
    ):
        raise HTTPException(
            status_code=422, detail="OpenCode expects one text user message"
        )
    return messages[0].content
