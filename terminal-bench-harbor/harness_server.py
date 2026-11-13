from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import uvicorn
from ac2.harness.types import (
    GradingStatus,
    GradingStatusResponse,
    RolloutRequest,
    RolloutStatus,
    ScoreResponse,
    StatusResponse,
    SubmitGradingRequest,
    SubmitPayload,
    SubmitResponse,
)
from fastapi import FastAPI, Header, HTTPException
from harbor.agents.installed.acp import AcpAgent
from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from harbor.trial.errors import AgentTimeoutError
from pydantic import BaseModel, ConfigDict, ValidationError

ACP_AGENT = "opencode@1.18.10"


class PinnedAcpAgent(AcpAgent):
    def _build_dependencies_command(self, kind: str) -> str:
        command = super()._build_dependencies_command(kind)
        unpinned = "pip install agent-client-protocol\n"
        assert unpinned in command
        return command.replace(unpinned, "pip install 'agent-client-protocol<1'\n")


class HarborDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ref: str
    task_name: str


class RolloutOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str | None = None
    trial_id: str | None = None
    trial_uri: str | None = None
    rewards: dict[str, float | int] | None = None
    error: str | None = None


@dataclass
class _Rollout:
    execution: asyncio.Task[JobResult]
    output: RolloutOutput | None = None


class TerminalBenchHarness:
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

        rollout_id = f"tbench_{uuid.uuid4().hex}"
        execution = asyncio.create_task(_run_harbor(rollout_id, request, dataset))
        self._rollouts[rollout_id] = _Rollout(execution=execution)
        self._submission_keys[idempotency_key] = rollout_id
        return rollout_id

    async def outputs(self, rollout_id: str) -> RolloutOutput:
        rollout = self.get(rollout_id)
        if rollout.output is not None:
            return rollout.output
        try:
            result = await asyncio.shield(rollout.execution)
        except Exception as exc:
            output = RolloutOutput(error=f"{type(exc).__name__}: {exc}")
        else:
            output = _output_from_result(result)
        rollout.output = output
        return output

    def get(self, rollout_id: str) -> _Rollout:
        rollout = self._rollouts.get(rollout_id)
        if rollout is None:
            raise HTTPException(status_code=404, detail="Unknown rollout")
        return rollout


def create_app(harness: TerminalBenchHarness) -> FastAPI:
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
        if not harness.get(request.rollout_id).execution.done():
            return StatusResponse(status=RolloutStatus.running)
        output = await harness.outputs(request.rollout_id)
        return StatusResponse(
            status=RolloutStatus.failed if output.error else RolloutStatus.completed,
            error_message=output.error and output.error[:2000],
        )

    @app.post("/collect_outputs", response_model=RolloutOutput | None)
    async def collect_outputs(request: RolloutRequest) -> RolloutOutput | None:
        output = await harness.outputs(request.rollout_id)
        return output if _reward(output) is not None else None

    @app.post("/submit_grading")
    async def submit_grading(request: SubmitGradingRequest) -> None:
        harness.get(request.rollout_id)

    @app.post("/get_grading_status", response_model=GradingStatusResponse)
    async def get_grading_status(request: RolloutRequest) -> GradingStatusResponse:
        output = await harness.outputs(request.rollout_id)
        if _reward(output) is None:
            return GradingStatusResponse(
                status=GradingStatus.failed,
                error_message=output.error and output.error[:2000],
            )
        return GradingStatusResponse(status=GradingStatus.completed)

    @app.post("/collect_score", response_model=ScoreResponse)
    async def collect_score(request: RolloutRequest) -> ScoreResponse:
        output = await harness.outputs(request.rollout_id)
        reward = _reward(output)
        if reward is None:
            raise HTTPException(status_code=409, detail="No Harbor reward")
        return ScoreResponse(
            score=reward,
            reasoning=f"Harbor verifier reward={reward}",
            artifacts={"rewards": output.rewards, "trial_uri": output.trial_uri},
        )

    return app


def _reward(output: RolloutOutput) -> float | None:
    reward = (output.rewards or {}).get("reward")
    if reward is None or not math.isfinite(reward):
        return None
    return reward


def _output_from_result(result: JobResult) -> RolloutOutput:
    if len(result.trial_results) != 1:
        return RolloutOutput(
            job_id=str(result.id),
            error=f"Harbor produced {len(result.trial_results)} trial results; expected one",
        )

    trial = result.trial_results[0]
    exception = trial.exception_info
    agent_failed = (
        exception is not None and exception.exception_type != AgentTimeoutError.__name__
    )
    output = RolloutOutput(
        job_id=str(result.id),
        trial_id=str(trial.id),
        trial_uri=trial.trial_uri,
        rewards=trial.verifier_result.rewards
        if trial.verifier_result is not None and not agent_failed
        else None,
        error=exception.exception_message if exception is not None else None,
    )
    if not output.error and _reward(output) is None:
        output.error = "Harbor verifier did not produce a reward"
    return output


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
            jobs_dir=Path(tempfile.gettempdir()) / "terminal-bench-harbor",
            n_concurrent_trials=1,
            quiet=True,
            agents=[
                AgentConfig(
                    import_path=f"{__name__}:PinnedAcpAgent",
                    kwargs={"registry_spec": ACP_AGENT},
                    env={
                        "AC2_ROLLOUT_API_KEY": request.ac_api_key,
                        "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config),
                    },
                )
            ],
            environment=EnvironmentConfig(type=EnvironmentType.MODAL),
            datasets=[
                DatasetConfig(
                    name=dataset.name,
                    ref=dataset.ref,
                    task_names=[dataset.task_name],
                )
            ],
        )
    )
    return await job.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(TerminalBenchHarness()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
