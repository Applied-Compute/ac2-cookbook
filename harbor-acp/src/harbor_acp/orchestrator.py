from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

from ac2.runtime import (
    BlackBoxOrchestratorProtocol,
    BlackBoxRolloutStatus,
    Input,
    Message,
)
from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from pydantic import JsonValue

from .environment import HarborDataset, HarborEnvironment

ACP_AGENT = "acp:opencode@1.18.10"
HARNESS_MODEL = "ac2/model"


class HarborACPOrchestrator(BlackBoxOrchestratorProtocol):
    agent_name = "opencode"
    POLLING_INTERVAL_S = 1

    def __init__(self) -> None:
        self.env = HarborEnvironment()
        self._rollouts: dict[str, asyncio.Task[JobResult]] = {}

    async def trigger_rollout(
        self,
        rollout_proxy_url: str,
        api_key: str,
        turn_input: Input,
        rollout_id: str | None = None,
    ) -> str:
        if rollout_id is not None:
            raise ValueError("This Harbor example only supports one user turn")
        if (
            len(turn_input) != 1
            or not isinstance(turn_input[0], Message)
            or turn_input[0].role != "user"
        ):
            raise ValueError("This Harbor example expects one user message")

        external_rollout_id = uuid.uuid4().hex
        self._rollouts[external_rollout_id] = asyncio.create_task(
            _run_harbor(
                rollout_id=external_rollout_id,
                dataset=self.env.require_dataset(),
                rollout_proxy_url=rollout_proxy_url,
                api_key=api_key,
            ),
            name=f"harbor-rollout-{external_rollout_id}",
        )
        return external_rollout_id

    async def check_status(self, ext_rollout_id: str) -> BlackBoxRolloutStatus:
        rollout = self._rollouts[ext_rollout_id]
        if not rollout.done():
            return BlackBoxRolloutStatus.RUNNING
        if rollout.cancelled() or rollout.exception() is not None:
            return BlackBoxRolloutStatus.ERRORED
        if rollout.result().stats.n_errored_trials:
            return BlackBoxRolloutStatus.ERRORED
        return BlackBoxRolloutStatus.COMPLETED

    async def fetch_assets(self, ext_rollout_id: str) -> JsonValue:
        result = await self._rollouts[ext_rollout_id]
        if len(result.trial_results) != 1:
            raise RuntimeError("Expected Harbor to produce exactly one trial result")

        trial = result.trial_results[0]
        self.env.trial_result = trial
        return {
            "job_id": str(result.id),
            "trial_id": str(trial.id),
            "trial_uri": trial.trial_uri,
            "rewards": trial.verifier_result.rewards
            if trial.verifier_result is not None
            else None,
            "error": trial.exception_info.exception_message
            if trial.exception_info is not None
            else None,
        }


async def _run_harbor(
    *,
    rollout_id: str,
    dataset: HarborDataset,
    rollout_proxy_url: str,
    api_key: str,
) -> JobResult:
    config = {
        "$schema": "https://opencode.ai/config.json",
        "agent": {"title": {"disable": True}},
        "provider": {
            "ac2": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "AC2 custom harness",
                "options": {
                    "apiKey": "{env:AC2_ROLLOUT_API_KEY}",
                    "baseURL": rollout_proxy_url,
                },
                "models": {"model": {"name": "Model selected by AC2"}},
            }
        },
    }
    job = await Job.create(
        JobConfig(
            job_name=rollout_id,
            jobs_dir=Path(tempfile.gettempdir()) / "harbor-acp-custom-harness",
            n_concurrent_trials=1,
            quiet=True,
            agents=[
                AgentConfig(
                    name=ACP_AGENT,
                    model_name=HARNESS_MODEL,
                    kwargs={"auth_policy": "auto", "permission_mode": "allow"},
                    env={
                        "AC2_ROLLOUT_API_KEY": api_key,
                        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                    },
                )
            ],
            environment=EnvironmentConfig(type=EnvironmentType.MODAL),
            datasets=[DatasetConfig(name=dataset.name, ref=dataset.ref)],
        )
    )
    return await job.run()
