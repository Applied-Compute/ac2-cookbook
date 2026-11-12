from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import uuid
from dataclasses import dataclass
from typing import Annotated, cast

import modal
import uvicorn
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
SANDBOX_TIMEOUT_SECONDS = 3 * 60 * 60
ROLLOUT_TIMEOUT_SECONDS = 2 * 60 * 60
GRADER_TIMEOUT_SECONDS = 240
OUTPUT_METADATA_LIMIT = 4_000


class SwebenchTaskParams(BaseModel):
    datapoint: dict[str, JsonValue]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Rollout:
    sandbox: modal.Sandbox
    image_head: str
    opencode: CommandResult


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
    if error := rollout.exception():
        return StatusResponse(
            status=RolloutStatus.failed, error_message=str(error)[:2000]
        )
    returncode = rollout.result().opencode.returncode
    if returncode != 0:
        return StatusResponse(
            status=RolloutStatus.failed,
            error_message=f"OpenCode exited with code {returncode}",
        )
    return StatusResponse(status=RolloutStatus.completed)


@app.post("/collect_outputs")
async def collect_outputs(request: RolloutRequest) -> dict[str, JsonValue]:
    try:
        rollout = await _rollout(request.rollout_id)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "harness_sandbox_id": rollout.sandbox.object_id,
        "returncode": rollout.opencode.returncode,
        "stdout": rollout.opencode.stdout[-OUTPUT_METADATA_LIMIT:],
        "stderr": rollout.opencode.stderr[-OUTPUT_METADATA_LIMIT:],
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
        head = await run_command(sandbox, "git rev-parse HEAD", timeout_seconds=60)
        if head.returncode != 0:
            raise RuntimeError(f"Failed to read image commit: {head.stderr}")

        setup = await run_command(sandbox, "mkdir -p /task", timeout_seconds=60)
        if setup.returncode != 0:
            raise RuntimeError(f"Failed to prepare harness sandbox: {setup.stderr}")
        await sandbox.filesystem.write_text.aio(instruction, "/task/instruction.md")

        config = {
            "$schema": "https://opencode.ai/config.json",
            "agent": {"title": {"disable": True}},
            "provider": {
                "ac2": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "AC2 custom harness",
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
            timeout_seconds=ROLLOUT_TIMEOUT_SECONDS,
        )
    except BaseException:
        await sandbox.terminate.aio()
        raise
    return Rollout(sandbox=sandbox, image_head=head.stdout.strip(), opencode=opencode)


async def grade(
    rollout_task: asyncio.Task[Rollout], datapoint: dict[str, JsonValue]
) -> ScoreResponse:
    rollout = await rollout_task
    test_spec = _test_spec(datapoint)
    sandboxes = [rollout.sandbox]
    try:
        patch_result = await run_command(
            rollout.sandbox,
            """
set -euo pipefail
git add --intent-to-add .
git diff --binary "$IMAGE_HEAD"
""",
            env={"IMAGE_HEAD": rollout.image_head},
            timeout_seconds=300,
        )
        if patch_result.returncode != 0:
            raise RuntimeError(f"Failed to fetch model patch: {patch_result.stderr}")

        sandbox = await create_sandbox(
            modal.Image.from_registry(test_spec.instance_image_key).entrypoint([])
        )
        sandboxes.append(sandbox)

        head_result = await run_command(
            sandbox, "git rev-parse HEAD", timeout_seconds=60
        )
        if head_result.returncode != 0:
            raise RuntimeError(
                f"Failed to verify grading sandbox image commit: {head_result.stderr.strip()}"
            )
        if head_result.stdout.strip() != rollout.image_head:
            raise RuntimeError(
                "Grading sandbox does not match the harness image commit: "
                f"expected {rollout.image_head}, got {head_result.stdout.strip()}"
            )
        status_result = await run_command(
            sandbox,
            "git status --porcelain --untracked-files=no",
            timeout_seconds=60,
        )
        if status_result.returncode != 0:
            raise RuntimeError(
                f"Failed to verify grading sandbox worktree: {status_result.stderr.strip()}"
            )
        if status_result.stdout.strip():
            raise RuntimeError(
                f"Grading sandbox has pre-existing tracked changes: {status_result.stdout.strip()}"
            )

        if patch_result.stdout:
            await sandbox.filesystem.write_text.aio(
                patch_result.stdout, "/tmp/model.patch"
            )
            check_result = await run_command(
                sandbox,
                "git apply --check --binary /tmp/model.patch",
                timeout_seconds=300,
            )
            if check_result.returncode != 0:
                patch_error = (
                    check_result.stderr.strip()
                    or f"git apply --check exited with code {check_result.returncode}"
                )
                return ScoreResponse(
                    score=0.0,
                    reasoning=(
                        "model patch could not be applied to the verified task base; "
                        "scored as an invalid submission"
                    ),
                    artifacts={
                        "resolved": False,
                        "patch_applied": False,
                        "patch_error": patch_error[:2_000],
                    },
                )

            apply_result = await run_command(
                sandbox,
                "git apply --binary /tmp/model.patch",
                timeout_seconds=300,
            )
            if apply_result.returncode != 0:
                raise RuntimeError(
                    "Patch passed git apply --check but failed during application: "
                    f"{apply_result.stderr}"
                )

        await sandbox.filesystem.write_text.aio(test_spec.eval_script, "/root/eval.sh")
        result = await run_command(
            sandbox,
            "cd /testbed && bash /root/eval.sh",
            timeout_seconds=GRADER_TIMEOUT_SECONDS,
        )
    finally:
        await asyncio.gather(*(sandbox.terminate.aio() for sandbox in sandboxes))

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
    resolved = bool(instance_report.get("resolved", False))
    tests_status = instance_report.get("tests_status", {})
    fail_to_pass = tests_status.get("FAIL_TO_PASS", {})
    pass_to_pass = tests_status.get("PASS_TO_PASS", {})
    fail_to_pass_succeeded = sorted(fail_to_pass.get("success", []))
    fail_to_pass_failed = sorted(fail_to_pass.get("failure", []))
    pass_to_pass_succeeded = sorted(pass_to_pass.get("success", []))
    pass_to_pass_failed = sorted(pass_to_pass.get("failure", []))

    return ScoreResponse(
        score=1.0 if resolved else 0.0,
        reasoning=(
            f"resolved={resolved}; "
            f"fail-to-pass={len(fail_to_pass_succeeded)}/"
            f"{len(fail_to_pass_succeeded) + len(fail_to_pass_failed)}; "
            f"pass-to-pass={len(pass_to_pass_succeeded)}/"
            f"{len(pass_to_pass_succeeded) + len(pass_to_pass_failed)}"
        ),
        artifacts={
            "resolved": resolved,
            "model_patch_bytes": len(patch_result.stdout.encode()),
            "fail_to_pass_succeeded": fail_to_pass_succeeded,
            "fail_to_pass_failed": fail_to_pass_failed,
            "pass_to_pass_succeeded": pass_to_pass_succeeded,
            "pass_to_pass_failed": pass_to_pass_failed,
            "grader_returncode": result.returncode,
        },
    )


async def create_sandbox(image: modal.Image) -> modal.Sandbox:
    modal_app = await modal.App.lookup.aio(
        "swebench-opencode-custom-harness", create_if_missing=True
    )
    with modal.enable_output():
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
    timeout_seconds: int = 0,
) -> CommandResult:
    proc = await sandbox.exec.aio(
        "bash",
        "-c",
        command,
        env=env,
        timeout=timeout_seconds or None,
        text=False,
    )
    await proc.wait.aio()
    stdout = (await proc.stdout.read.aio()).decode("utf-8", errors="replace")
    stderr = (await proc.stderr.read.aio()).decode("utf-8", errors="replace")
    if proc.returncode is None:
        raise RuntimeError("Sandbox process ended without an exit code")
    return CommandResult(returncode=proc.returncode, stdout=stdout, stderr=stderr)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
