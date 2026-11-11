from __future__ import annotations

import asyncio
import uuid
from typing import Annotated

from ac2.harness.types import (
    RolloutRequest,
    RolloutStatus,
    StatusResponse,
    SubmitPayload,
)
from ac2.runtime import Message
from fastapi import FastAPI, Header, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from byoh_smoke_example.models import HarnessOutput

app = FastAPI()
rollouts: dict[str, asyncio.Task[HarnessOutput]] = {}
rollout_ids_by_key: dict[str, str] = {}


class SubmitResponse(BaseModel):
    rollout_id: str


async def run_rollout(request: SubmitPayload) -> HarnessOutput:
    messages = [Message.model_validate(item) for item in request.task.input]
    async with AsyncOpenAI(
        api_key=request.ac_api_key, base_url=request.ac_api_url
    ) as model:
        completion = await model.chat.completions.create(
            model="policy",
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
    return HarnessOutput(answer=completion.choices[0].message.content or "")


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
        return StatusResponse(status=RolloutStatus.failed)
    return StatusResponse(status=RolloutStatus.completed)


@app.post("/collect_outputs", response_model=HarnessOutput)
async def collect_outputs(request: RolloutRequest) -> HarnessOutput:
    rollout = _rollout(request.rollout_id)
    try:
        return await rollout
    except Exception as exc:
        return HarnessOutput(answer="", error=f"{type(exc).__name__}: {exc}")


def _rollout(rollout_id: str) -> asyncio.Task[HarnessOutput]:
    rollout = rollouts.get(rollout_id)
    if rollout is None:
        raise HTTPException(status_code=404, detail="Unknown rollout")
    return rollout
