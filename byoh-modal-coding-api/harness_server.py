from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import uvicorn
from ac2.harness.types import (
    HarnessTask,
    RolloutRequest,
    RolloutStatus,
    StatusResponse,
    SubmitPayload,
)
from ac2.runtime import Message
from fastapi import FastAPI, Header, HTTPException
from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from pydantic import BaseModel, ValidationError

from byoh_modal_coding_api.models import CodingRolloutOutput, HarborDataset

ACP_AGENT = "acp:opencode@1.18.10"


class SubmitResponse(BaseModel):
    rollout_id: str


@dataclass
class _Rollout:
    execution: asyncio.Task[JobResult]
    output: CodingRolloutOutput | None = None


class ModalCodingHarness:
    def __init__(self) -> None:
        self._rollouts: dict[str, _Rollout] = {}
        self._submission_keys: dict[str, str] = {}

    def submit(self, request: SubmitPayload, idempotency_key: str) -> str:
        existing_id = self._submission_keys.get(idempotency_key)
        if existing_id is not None:
            return existing_id

        try:
            dataset = HarborDataset.model_validate(request.task.env_params or {})
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail="Invalid Harbor dataset configuration"
            ) from exc
        _validate_prompt(request.task)

        rollout_id = f"coding_{uuid.uuid4().hex}"
        execution = asyncio.create_task(
            _run_harbor(rollout_id, request, dataset),
            name=f"modal-coding-{rollout_id}",
        )
        self._rollouts[rollout_id] = _Rollout(execution=execution)
        self._submission_keys[idempotency_key] = rollout_id
        return rollout_id

    def status(self, rollout_id: str) -> RolloutStatus:
        rollout = self._get(rollout_id)
        if rollout.output is not None:
            return (
                RolloutStatus.failed
                if rollout.output.error
                else RolloutStatus.completed
            )
        if not rollout.execution.done():
            return RolloutStatus.running
        if rollout.execution.cancelled() or rollout.execution.exception() is not None:
            return RolloutStatus.failed
        if rollout.execution.result().stats.n_errored_trials:
            return RolloutStatus.failed
        return RolloutStatus.completed

    async def outputs(self, rollout_id: str) -> CodingRolloutOutput:
        rollout = self._get(rollout_id)
        if rollout.output is not None:
            return rollout.output
        try:
            result = await asyncio.shield(rollout.execution)
        except asyncio.CancelledError:
            if not rollout.execution.cancelled():
                raise
            output = CodingRolloutOutput(error="Harbor rollout was cancelled")
        except Exception as exc:
            output = CodingRolloutOutput(error=f"{type(exc).__name__}: {exc}")
        else:
            output = _output_from_result(result)
        rollout.output = output
        return output

    def _get(self, rollout_id: str) -> _Rollout:
        rollout = self._rollouts.get(rollout_id)
        if rollout is None:
            raise HTTPException(status_code=404, detail="Unknown rollout")
        return rollout


def create_app(harness: ModalCodingHarness) -> FastAPI:
    app = FastAPI()

    @app.post("/submit", response_model=SubmitResponse)
    async def submit(
        request: SubmitPayload,
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=1, max_length=255)
        ],
    ) -> SubmitResponse:
        return SubmitResponse(rollout_id=harness.submit(request, idempotency_key))

    @app.post("/get_status", response_model=StatusResponse)
    async def get_status(request: RolloutRequest) -> StatusResponse:
        return StatusResponse(status=harness.status(request.rollout_id))

    @app.post("/collect_outputs", response_model=CodingRolloutOutput)
    async def collect_outputs(request: RolloutRequest) -> CodingRolloutOutput:
        return await harness.outputs(request.rollout_id)

    return app


def _validate_prompt(task: HarnessTask) -> None:
    if len(task.input) != 1:
        raise HTTPException(status_code=422, detail="Expected one input message")
    try:
        message = Message.model_validate(task.input[0])
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail="Expected one text user message"
        ) from exc
    if message.role != "user" or not isinstance(message.content, str):
        raise HTTPException(status_code=422, detail="Expected one text user message")


def _output_from_result(result: JobResult) -> CodingRolloutOutput:
    if len(result.trial_results) != 1:
        return CodingRolloutOutput(
            job_id=str(result.id),
            error=f"Harbor produced {len(result.trial_results)} trial results; expected one",
        )

    trial = result.trial_results[0]
    return CodingRolloutOutput(
        job_id=str(result.id),
        trial_id=str(trial.id),
        trial_uri=trial.trial_uri,
        rewards=trial.verifier_result.rewards
        if trial.verifier_result is not None
        else None,
        error=trial.exception_info.exception_message
        if trial.exception_info is not None
        else None,
    )


async def _run_harbor(
    rollout_id: str, request: SubmitPayload, dataset: HarborDataset
) -> JobResult:
    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "agent": {"title": {"disable": True}},
        "model": "ac2/model",
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
    job = await Job.create(
        JobConfig(
            job_name=rollout_id,
            jobs_dir=Path(tempfile.gettempdir()) / "byoh-modal-coding-api",
            n_concurrent_trials=1,
            quiet=True,
            agents=[
                AgentConfig(
                    name=ACP_AGENT,
                    kwargs={"auth_policy": "auto", "permission_mode": "allow"},
                    env={
                        "AC2_ROLLOUT_API_KEY": request.ac_api_key,
                        "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config),
                    },
                )
            ],
            environment=EnvironmentConfig(type=EnvironmentType.MODAL),
            datasets=[DatasetConfig(name=dataset.name, ref=dataset.ref)],
        )
    )
    return await job.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(ModalCodingHarness()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
