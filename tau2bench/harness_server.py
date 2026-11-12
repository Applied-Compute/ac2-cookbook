from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Annotated, Any

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
from ac2.runtime import Environment, FunctionCall
from fastapi import FastAPI, Header, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

from tau2bench.dataloader import load_tau2_policy
from tau2bench.environments import (
    AirlineEnvironment,
    RetailEnvironment,
    TelecomEnvironment,
)
from tau2bench.graders import grade
from tau2bench.user import USER_SIMULATOR_ERROR, Tau2BenchUser

MAX_STEPS = 200

DOMAIN_ENVIRONMENT: dict[str, type[Environment]] = {
    "airline": AirlineEnvironment,
    "retail": RetailEnvironment,
    "telecom": TelecomEnvironment,
}


class RolloutOutput(BaseModel):
    num_messages: int
    termination_reason: str | None = None
    error: str | None = None


@dataclass
class Rollout:
    env: Environment | None
    messages: list[dict[str, Any]]
    conversation: asyncio.Task[None]
    grading: asyncio.Task[ScoreResponse] | None = None


app = FastAPI()
openai_client = AsyncOpenAI()
rollouts: dict[str, Rollout] = {}
rollout_ids_by_key: dict[str, str] = {}


async def run_conversation(
    request: SubmitPayload, env: Environment, messages: list[dict[str, Any]]
) -> None:
    env_params = request.task.env_params or {}
    await env.setup(env_params)
    user = Tau2BenchUser(env_params["user_instructions"], openai_client)
    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in env.tools
    ]
    messages.append({"role": "system", "content": load_tau2_policy(env.domain)})

    async with AsyncOpenAI(
        api_key=request.ac_api_key, base_url=request.ac_api_url
    ) as model:
        while not env.terminated:
            if len(messages) > MAX_STEPS:
                env.terminated = True
                env.termination_reason = "max_steps"
                break

            last = messages[-1]
            if last["role"] != "tool":
                if last["role"] == "assistant" and not last["content"]:
                    break
                try:
                    customer_text = await user.respond(
                        last["content"] if last["role"] == "assistant" else None
                    )
                except Exception:
                    env.terminated = True
                    env.termination_reason = USER_SIMULATOR_ERROR
                    break
                if customer_text is None:
                    break
                messages.append({"role": "user", "content": customer_text})

            completion = await model.chat.completions.create(
                model="policy", messages=messages, tools=tools
            )
            reply = completion.choices[0].message
            calls = reply.tool_calls or []
            assistant: dict[str, Any] = {"role": "assistant", "content": reply.content}
            if calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in calls
                ]
            messages.append(assistant)
            for call in calls:
                result = await env.call_tool(
                    FunctionCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=call.function.arguments,
                    )
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result.output}
                )


async def run_grading(request: SubmitGradingRequest, rollout: Rollout) -> ScoreResponse:
    if request.rollout_status is RolloutStatus.failed:
        return ScoreResponse(score=0, reasoning="The rollout failed before it finished.")
    return await grade(
        request.task.grader_params or {}, rollout.messages, rollout.env, openai_client
    )


@app.post("/submit", response_model=SubmitResponse)
async def submit(
    request: SubmitPayload,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> SubmitResponse:
    if idempotency_key not in rollout_ids_by_key:
        env_params = request.task.env_params or {}
        env_class = DOMAIN_ENVIRONMENT.get(str(env_params.get("domain")))
        if env_class is None or "user_instructions" not in env_params:
            raise HTTPException(
                status_code=422,
                detail="env_params needs a tau2-bench domain and user_instructions",
            )
        env = env_class()
        messages: list[dict[str, Any]] = []
        rollout_id = f"tau2_{uuid.uuid4().hex}"
        rollouts[rollout_id] = Rollout(
            env=env,
            messages=messages,
            conversation=asyncio.create_task(run_conversation(request, env, messages)),
        )
        rollout_ids_by_key[idempotency_key] = rollout_id
    return SubmitResponse(rollout_id=rollout_ids_by_key[idempotency_key])


@app.post("/get_status", response_model=StatusResponse)
async def get_status(request: RolloutRequest) -> StatusResponse:
    conversation = _rollout(request.rollout_id).conversation
    if not conversation.done():
        return StatusResponse(status=RolloutStatus.running)
    if conversation.exception() is not None:
        return StatusResponse(
            status=RolloutStatus.failed, error_message=_error(conversation)
        )
    return StatusResponse(status=RolloutStatus.completed)


@app.post("/collect_outputs", response_model=RolloutOutput)
async def collect_outputs(request: RolloutRequest) -> RolloutOutput:
    rollout = _rollout(request.rollout_id)
    return RolloutOutput(
        num_messages=len(rollout.messages),
        termination_reason=getattr(rollout.env, "termination_reason", None),
        error=_error(rollout.conversation),
    )


@app.post("/submit_grading")
async def submit_grading(request: SubmitGradingRequest) -> dict[str, str]:
    rollout = _rollout(request.rollout_id)
    if rollout.grading is None:
        rollout.grading = asyncio.create_task(run_grading(request, rollout))
    return {}


@app.post("/get_grading_status", response_model=GradingStatusResponse)
async def get_grading_status(request: RolloutRequest) -> GradingStatusResponse:
    grading = _grading(request.rollout_id)
    if not grading.done():
        return GradingStatusResponse(status=GradingStatus.running)
    if grading.exception() is not None:
        return GradingStatusResponse(
            status=GradingStatus.failed, error_message=_error(grading)
        )
    return GradingStatusResponse(status=GradingStatus.completed)


@app.post("/collect_score", response_model=ScoreResponse)
async def collect_score(request: RolloutRequest) -> ScoreResponse:
    score = await _grading(request.rollout_id)
    _rollout(request.rollout_id).env = None
    return score


def _rollout(rollout_id: str) -> Rollout:
    rollout = rollouts.get(rollout_id)
    if rollout is None:
        raise HTTPException(status_code=404, detail="Unknown rollout")
    return rollout


def _grading(rollout_id: str) -> asyncio.Task[ScoreResponse]:
    grading = _rollout(rollout_id).grading
    if grading is None:
        raise HTTPException(status_code=404, detail="Grading was not submitted")
    return grading


def _error(task: asyncio.Task[Any]) -> str | None:
    if not task.done() or task.exception() is None:
        return None
    exc = task.exception()
    return f"{type(exc).__name__}: {exc}"[:2000]
