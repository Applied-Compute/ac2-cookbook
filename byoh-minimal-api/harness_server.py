from __future__ import annotations

import argparse
import asyncio
import uuid
from dataclasses import dataclass
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
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from byoh_minimal_api.models import MinimalOutput


class SubmitResponse(BaseModel):
    rollout_id: str


@dataclass
class _Rollout:
    execution: asyncio.Task[MinimalOutput]
    output: MinimalOutput | None = None


class MinimalHarness:
    def __init__(self) -> None:
        self._rollouts: dict[str, _Rollout] = {}
        self._submission_keys: dict[str, str] = {}

    def submit(self, request: SubmitPayload, idempotency_key: str) -> str:
        existing_id = self._submission_keys.get(idempotency_key)
        if existing_id is not None:
            return existing_id

        _question(request.task)
        rollout_id = f"minimal_{uuid.uuid4().hex}"
        execution = asyncio.create_task(
            run_rollout(request), name=f"minimal-{rollout_id}"
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
        return RolloutStatus.completed

    async def outputs(self, rollout_id: str) -> MinimalOutput:
        rollout = self._get(rollout_id)
        if rollout.output is not None:
            return rollout.output
        try:
            output = await asyncio.shield(rollout.execution)
        except asyncio.CancelledError:
            if not rollout.execution.cancelled():
                raise
            output = MinimalOutput(answer="", error="Rollout was cancelled")
        except Exception as exc:
            output = MinimalOutput(answer="", error=f"{type(exc).__name__}: {exc}")
        rollout.output = output
        return output

    def _get(self, rollout_id: str) -> _Rollout:
        rollout = self._rollouts.get(rollout_id)
        if rollout is None:
            raise HTTPException(status_code=404, detail="Unknown rollout")
        return rollout


def create_app(harness: MinimalHarness) -> FastAPI:
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

    @app.post("/collect_outputs", response_model=MinimalOutput)
    async def collect_outputs(request: RolloutRequest) -> MinimalOutput:
        return await harness.outputs(request.rollout_id)

    return app


async def run_rollout(request: SubmitPayload) -> MinimalOutput:
    async with AsyncOpenAI(
        api_key=request.ac_api_key, base_url=request.ac_api_url
    ) as model:
        completion = await model.chat.completions.create(
            model="policy",
            messages=[{"role": "user", "content": _question(request.task)}],
        )
    return MinimalOutput(answer=completion.choices[0].message.content or "")


def _question(task: HarnessTask) -> str:
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
    return message.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(MinimalHarness()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
