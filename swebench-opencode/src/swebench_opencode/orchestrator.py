from __future__ import annotations

import asyncio
import json
import uuid

from pydantic import JsonValue

from ac2.runtime import (
    BlackBoxOrchestratorProtocol,
    BlackBoxRolloutStatus,
    Input,
    Message,
)

from .environment import SwebenchOpenCodeEnvironment
from .modal_utils import CommandResult, run_command

OPENCODE = "/opt/opencode/opencode"
ROLLOUT_TIMEOUT_SECONDS = 2 * 60 * 60
OUTPUT_METADATA_LIMIT = 4_000


class SwebenchOpenCodeOrchestrator(BlackBoxOrchestratorProtocol):
    agent_name = "opencode"
    POLLING_INTERVAL_S = 1
    ROLLOUT_TIMEOUT_S = ROLLOUT_TIMEOUT_SECONDS + 5 * 60

    def __init__(self) -> None:
        self.env = SwebenchOpenCodeEnvironment()
        self._rollouts: dict[str, asyncio.Task[CommandResult]] = {}

    async def trigger_rollout(
        self,
        rollout_proxy_url: str,
        api_key: str,
        turn_input: Input,
        rollout_id: str | None = None,
    ) -> str:
        if rollout_id is not None:
            raise ValueError("OpenCode SWE-bench rollouts only support one turn")
        external_rollout_id = uuid.uuid4().hex
        self._rollouts[external_rollout_id] = asyncio.create_task(
            self._run_opencode(
                rollout_proxy_url,
                api_key,
                _single_user_prompt(turn_input),
            ),
            name=f"opencode-rollout-{external_rollout_id}",
        )
        return external_rollout_id

    async def check_status(self, ext_rollout_id: str) -> BlackBoxRolloutStatus:
        rollout = self._rollouts[ext_rollout_id]
        if not rollout.done():
            return BlackBoxRolloutStatus.RUNNING
        if rollout.cancelled() or rollout.exception() is not None:
            return BlackBoxRolloutStatus.ERRORED
        if rollout.result().returncode != 0:
            return BlackBoxRolloutStatus.ERRORED
        return BlackBoxRolloutStatus.COMPLETED

    async def fetch_assets(self, ext_rollout_id: str) -> JsonValue:
        result = await self._rollouts[ext_rollout_id]
        model_patch = await self.env.materialize_grading_sandbox()
        return {
            "harness_sandbox_id": self.env.harness_sandbox.object_id,
            "grading_sandbox_id": self.env.sandbox.object_id,
            "model_patch_bytes": len(model_patch.encode()),
            "returncode": result.returncode,
            "stdout": result.stdout[-OUTPUT_METADATA_LIMIT:],
            "stderr": result.stderr[-OUTPUT_METADATA_LIMIT:],
        }

    async def _run_opencode(
        self,
        rollout_proxy_url: str,
        api_key: str,
        prompt: str,
    ) -> CommandResult:
        config = {
            "$schema": "https://opencode.ai/config.json",
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
        return await run_command(
            self.env.harness_sandbox,
            (
                f"{OPENCODE} run --model ac2/model --format json "
                '--dangerously-skip-permissions "$OPENCODE_PROMPT" </dev/null'
            ),
            env={
                "AC2_ROLLOUT_API_KEY": api_key,
                "OPENCODE_PROMPT": prompt,
                "OPENCODE_CONFIG_CONTENT": json.dumps(config),
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
            },
            timeout_seconds=ROLLOUT_TIMEOUT_SECONDS,
        )


def _single_user_prompt(turn_input: Input) -> str:
    messages = [
        item
        for item in turn_input
        if isinstance(item, Message) and item.role == "user"
    ]
    if len(messages) != 1 or not isinstance(messages[0].content, str):
        raise ValueError("OpenCode expects one text user message")
    return messages[0].content
