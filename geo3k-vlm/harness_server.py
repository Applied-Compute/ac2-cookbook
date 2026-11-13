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
from math_verify import LatexExtractionConfig, parse, verify
from openai import AsyncOpenAI
from pydantic import BaseModel

SYSTEM_PROMPT = (
    "Solve the geometry problem using the image. Show your reasoning, "
    r"then put your final answer in \boxed{...}."
)

app = FastAPI()
rollouts: dict[str, asyncio.Task[RolloutOutput]] = {}
rollout_ids_by_key: dict[str, str] = {}
scores: dict[str, ScoreResponse] = {}


class RolloutOutput(BaseModel):
    answer: str
    finish_reason: str


async def run_rollout(request: SubmitPayload) -> RolloutOutput:
    messages = [Message.model_validate(item) for item in request.task.input]
    async with AsyncOpenAI(
        api_key=request.ac_api_key, base_url=request.ac_api_url
    ) as model:
        completion = await model.chat.completions.create(
            model="policy",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
        )
    choice = completion.choices[0]
    return RolloutOutput(
        answer=choice.message.content or "", finish_reason=choice.finish_reason
    )


def answer_score(response: str, expected: str) -> float:
    gold = parse(f"${expected}$", extraction_config=[LatexExtractionConfig()])
    predicted = parse(response, extraction_config=[LatexExtractionConfig()])
    return float(bool(gold and predicted and verify(gold, predicted)))


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
    if (error := rollout.exception()) is not None:
        return StatusResponse(
            status=RolloutStatus.failed,
            error_message=f"{type(error).__name__}: {error}"[:2000],
        )
    if rollout.result().finish_reason == "length":
        return StatusResponse(status=RolloutStatus.truncated)
    return StatusResponse(status=RolloutStatus.completed)


@app.post("/collect_outputs", response_model=RolloutOutput | None)
async def collect_outputs(request: RolloutRequest) -> RolloutOutput | None:
    rollout = _rollout(request.rollout_id)
    if rollout.exception() is not None:
        return None
    return rollout.result()


@app.post("/submit_grading")
async def submit_grading(request: SubmitGradingRequest) -> None:
    rollout = _rollout(request.rollout_id)
    if rollout.exception() is not None:
        return
    result = rollout.result()
    if result.finish_reason == "length":
        scores[request.rollout_id] = ScoreResponse(
            score=0.0, reasoning="Response reached the length cap."
        )
        return
    expected = str((request.task.grader_params or {})["answer"])
    score = answer_score(result.answer, expected)
    scores[request.rollout_id] = ScoreResponse(
        score=score,
        reasoning="Final answer matches." if score else "Final answer does not match.",
    )


@app.post("/get_grading_status", response_model=GradingStatusResponse)
async def get_grading_status(request: RolloutRequest) -> GradingStatusResponse:
    rollout = _rollout(request.rollout_id)
    if rollout.done() and (error := rollout.exception()) is not None:
        return GradingStatusResponse(
            status=GradingStatus.failed,
            error_message=f"{type(error).__name__}: {error}"[:2000],
        )
    if request.rollout_id not in scores:
        raise HTTPException(status_code=404, detail="Unknown grading")
    return GradingStatusResponse(status=GradingStatus.completed)


@app.post("/collect_score", response_model=ScoreResponse)
async def collect_score(request: RolloutRequest) -> ScoreResponse:
    if request.rollout_id not in scores:
        raise HTTPException(status_code=404, detail="Unknown grading")
    return scores[request.rollout_id]


def _rollout(rollout_id: str) -> asyncio.Task[RolloutOutput]:
    rollout = rollouts.get(rollout_id)
    if rollout is None:
        raise HTTPException(status_code=404, detail="Unknown rollout")
    return rollout
