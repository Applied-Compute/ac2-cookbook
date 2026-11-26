from __future__ import annotations

import asyncio
import uuid

from ac2.harness import (
    GradingStatus,
    GradingStatusResponse,
    Harness,
    RolloutStatus,
    ScoreResponse,
    StatusResponse,
    SubmitGradingRequest,
    SubmitRequest,
    SubmitResponse,
    UnknownRolloutError,
)
from ac2.runtime import Message
from openai import AsyncOpenAI


class SmokeHarness(Harness):
    """Asks the model each task's question and scores whether the answer contains `expected_answer`."""

    def __init__(self) -> None:
        self._rollouts: dict[str, asyncio.Task[str]] = {}
        self._scores: dict[str, ScoreResponse] = {}

    async def submit(self, request: SubmitRequest) -> SubmitResponse:
        rollout_id = f"rollout_{uuid.uuid4().hex}"
        self._rollouts[rollout_id] = asyncio.create_task(run_rollout(request))
        return SubmitResponse(rollout_id=rollout_id)

    async def get_status(self, rollout_id: str) -> StatusResponse:
        rollout = self._rollout(rollout_id)
        if not rollout.done():
            return StatusResponse(status=RolloutStatus.running)
        if rollout.exception() is not None:
            return StatusResponse(status=RolloutStatus.failed, error_message=_error_message(rollout))
        return StatusResponse(status=RolloutStatus.completed)

    async def collect_outputs(self, rollout_id: str) -> dict[str, str] | None:
        rollout = self._rollout(rollout_id)
        if rollout.exception() is not None:
            return None
        return {"answer": rollout.result()}

    async def submit_grading(self, request: SubmitGradingRequest) -> None:
        rollout = self._rollout(request.rollout_id)
        if rollout.exception() is not None:
            return
        expected_answer = str((request.task.grader_params or {})["expected_answer"])
        matched = expected_answer.casefold() in rollout.result().casefold()
        self._scores[request.rollout_id] = ScoreResponse(
            score=1 if matched else 0,
            reasoning=f"Expected the answer to contain {expected_answer!r}.",
        )

    async def get_grading_status(self, rollout_id: str) -> GradingStatusResponse:
        if rollout_id in self._scores:
            return GradingStatusResponse(status=GradingStatus.completed)
        rollout = self._rollout(rollout_id)
        if not rollout.done() or rollout.exception() is None:
            raise UnknownRolloutError(f"Rollout {rollout_id} is not graded")
        return GradingStatusResponse(status=GradingStatus.failed, error_message=_error_message(rollout))

    async def collect_score(self, rollout_id: str) -> ScoreResponse:
        if rollout_id not in self._scores:
            raise UnknownRolloutError(f"Rollout {rollout_id} has no score")
        return self._scores[rollout_id]

    def _rollout(self, rollout_id: str) -> asyncio.Task[str]:
        if rollout_id not in self._rollouts:
            raise UnknownRolloutError(f"Unknown rollout {rollout_id}")
        return self._rollouts[rollout_id]


async def run_rollout(request: SubmitRequest) -> str:
    messages = [Message.model_validate(item) for item in request.task.input]
    async with AsyncOpenAI(api_key=request.ac_api_key, base_url=request.ac_api_url) as model:
        completion = await model.chat.completions.create(
            model="policy",
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
    return completion.choices[0].message.content or ""


def _error_message(rollout: asyncio.Task[str]) -> str:
    exc = rollout.exception()
    return f"{type(exc).__name__}: {exc}"[:2000]
