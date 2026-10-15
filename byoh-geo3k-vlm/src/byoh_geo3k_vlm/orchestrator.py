import asyncio
import uuid

import httpx
from ac2.runtime import BlackBoxRolloutStatus, CustomHarnessOrchestrator, Input, Message
from openai.types.chat import ChatCompletion
from pydantic import JsonValue

COMPLETION_TIMEOUT_S = 840
MAX_RESPONSE_TOKENS = 8192


class Geo3KByohOrchestrator(CustomHarnessOrchestrator):
    POLLING_INTERVAL_S = 1
    ROLLOUT_TIMEOUT_S = 900

    def __init__(self) -> None:
        self._rollouts: dict[str, asyncio.Task[ChatCompletion]] = {}

    async def submit(
        self,
        rollout_proxy_url: str,
        api_key: str,
        turn_input: Input,
        continue_rollout_id: str | None = None,
    ) -> str:
        if continue_rollout_id is not None:
            raise ValueError("This harness supports one user turn per rollout")
        if len(turn_input) != 1 or not isinstance(turn_input[0], Message):
            raise ValueError("Expected one user message containing a geometry question and image")
        message = turn_input[0]
        if message.role != "user" or not isinstance(message.content, list):
            raise ValueError("Expected structured user content")
        if not any(part.get("type") == "image_url" for part in message.content):
            raise ValueError("Geometry3K tasks require an image")
        rollout_id = uuid.uuid4().hex
        self._rollouts[rollout_id] = asyncio.create_task(self._complete(rollout_proxy_url, api_key, message))
        return rollout_id

    async def get_status(self, rollout_id: str) -> BlackBoxRolloutStatus:
        rollout = self._rollouts[rollout_id]
        if not rollout.done():
            return BlackBoxRolloutStatus.RUNNING
        if rollout.cancelled() or rollout.exception() is not None:
            return BlackBoxRolloutStatus.ERRORED
        return BlackBoxRolloutStatus.COMPLETED

    async def collect_outputs(self, rollout_id: str) -> JsonValue:
        response = await self._rollouts.pop(rollout_id)
        choice = response.choices[0]
        return {"answer": choice.message.content, "finish_reason": choice.finish_reason}

    async def teardown(self) -> None:
        for rollout in self._rollouts.values():
            rollout.cancel()
        # Status and output collection report failures; teardown drains remaining tasks.
        await asyncio.gather(*self._rollouts.values(), return_exceptions=True)
        self._rollouts.clear()
        await super().teardown()

    async def _complete(self, base_url: str, api_key: str, message: Message) -> ChatCompletion:
        async with asyncio.timeout(COMPLETION_TIMEOUT_S):
            async with httpx.AsyncClient(timeout=COMPLETION_TIMEOUT_S, follow_redirects=True) as client:
                response = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "model-selected-by-ac2-job",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Solve the geometry problem using the image. Show your reasoning, "
                                r"then put your final answer in \boxed{...}.",
                            },
                            message.model_dump(include={"role", "content"}),
                        ],
                        "max_tokens": MAX_RESPONSE_TOKENS,
                        "temperature": 1.0,
                    },
                )
                response.raise_for_status()
                return ChatCompletion.model_validate_json(response.content)
