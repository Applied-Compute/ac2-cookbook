from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

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
from ac2.runtime import Message
from fastapi import FastAPI, Header, HTTPException
from openai import AsyncOpenAI

app = FastAPI()
rollouts: dict[str, asyncio.Task[str]] = {}
rollout_ids_by_key: dict[str, str] = {}
scores: dict[str, ScoreResponse] = {}


async def run_rollout(request: SubmitPayload) -> str:
    messages = [Message.model_validate(item) for item in request.task.input]
    async with AsyncOpenAI(
        api_key=request.ac_api_key, base_url=request.ac_api_url
    ) as model:
        completion = await model.chat.completions.create(
            model="policy",
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
    return completion.choices[0].message.content or ""


@app.post("/submit", response_model=SubmitResponse)
async def submit(
    request: SubmitPayload,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> SubmitResponse:
    if idempotency_key not in rollout_ids_by_key:
        rollout_id = f"rollout_{uuid.uuid4().hex}"
        rollouts[rollout_id] = asyncio.create_task(run_rollout(request))
        rollout_ids_by_key[idempotency_key] = rollout_id
    return SubmitResponse(rollout_id=rollout_ids_by_key[idempotency_key])


@app.post("/get_status", response_model=StatusResponse)
async def get_status(request: RolloutRequest) -> StatusResponse:
    rollout = _rollout(request.rollout_id)
    if not rollout.done():
        return StatusResponse(status=RolloutStatus.running)
    if rollout.exception() is not None:
        return StatusResponse(
            status=RolloutStatus.failed, error_message=_error_message(rollout)
        )
    return StatusResponse(status=RolloutStatus.completed)


@app.post("/collect_outputs")
async def collect_outputs(request: RolloutRequest) -> dict[str, str] | None:
    rollout = _rollout(request.rollout_id)
    if rollout.exception() is not None:
        return None
    return {"answer": rollout.result()}


@app.post("/submit_grading")
async def submit_grading(request: SubmitGradingRequest) -> None:
    rollout = _rollout(request.rollout_id)
    if rollout.exception() is not None:
        return
    expected_answer = str((request.task.grader_params or {})["expected_answer"])
    matched = expected_answer.casefold() in rollout.result().casefold()
    scores[request.rollout_id] = ScoreResponse(
        score=1 if matched else 0,
        reasoning=f"Expected the answer to contain {expected_answer!r}.",
    )


@app.post("/get_grading_status", response_model=GradingStatusResponse)
async def get_grading_status(request: RolloutRequest) -> GradingStatusResponse:
    if request.rollout_id in scores:
        return GradingStatusResponse(status=GradingStatus.completed)
    rollout = _rollout(request.rollout_id)
    if not rollout.done() or rollout.exception() is None:
        raise HTTPException(status_code=404, detail="Not graded")
    return GradingStatusResponse(
        status=GradingStatus.failed, error_message=_error_message(rollout)
    )


@app.post("/collect_score", response_model=ScoreResponse)
async def collect_score(request: RolloutRequest) -> ScoreResponse:
    if request.rollout_id not in scores:
        raise HTTPException(status_code=404, detail="No score")
    return scores[request.rollout_id]


def _error_message(rollout: asyncio.Task[str]) -> str:
    exc = rollout.exception()
    return f"{type(exc).__name__}: {exc}"[:2000]


def _rollout(rollout_id: str) -> asyncio.Task[str]:
    rollout = rollouts.get(rollout_id)
    if rollout is None:
        raise HTTPException(status_code=404, detail="Unknown rollout")
    return rollout
