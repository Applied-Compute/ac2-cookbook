from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import uuid
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import quote

import httpx
import uvicorn
from ac2.harness.types import (
    HarnessTask,
    RolloutRequest,
    RolloutStatus,
    StatusResponse,
    SubmitPayload,
)
from ac2.runtime import Message
from agents import Agent, OpenAIResponsesModel, RunConfig, Runner, function_tool
from fastapi import FastAPI, Header, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from byoh_agents_web_search_api.models import WebSearchOutput

_HTML_TAG = re.compile(r"<[^>]+>")


class SubmitResponse(BaseModel):
    rollout_id: str


class _WikipediaHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    snippet: str


class _WikipediaQuery(BaseModel):
    search: list[_WikipediaHit]


class _WikipediaResponse(BaseModel):
    query: _WikipediaQuery


@dataclass
class _Rollout:
    execution: asyncio.Task[WebSearchOutput]
    output: WebSearchOutput | None = None


class WebSearchHarness:
    def __init__(self) -> None:
        self._rollouts: dict[str, _Rollout] = {}
        self._submission_keys: dict[str, str] = {}

    def submit(self, request: SubmitPayload, idempotency_key: str) -> str:
        existing_id = self._submission_keys.get(idempotency_key)
        if existing_id is not None:
            return existing_id

        _question(request.task)
        rollout_id = f"web_{uuid.uuid4().hex}"
        execution = asyncio.create_task(
            run_web_search(request), name=f"web-search-{rollout_id}"
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

    async def outputs(self, rollout_id: str) -> WebSearchOutput:
        rollout = self._get(rollout_id)
        if rollout.output is not None:
            return rollout.output
        try:
            output = await asyncio.shield(rollout.execution)
        except asyncio.CancelledError:
            if not rollout.execution.cancelled():
                raise
            output = WebSearchOutput(
                answer="", error="Web-search rollout was cancelled"
            )
        except Exception as exc:
            output = WebSearchOutput(answer="", error=f"{type(exc).__name__}: {exc}")
        rollout.output = output
        return output

    def _get(self, rollout_id: str) -> _Rollout:
        rollout = self._rollouts.get(rollout_id)
        if rollout is None:
            raise HTTPException(status_code=404, detail="Unknown rollout")
        return rollout


def create_app(harness: WebSearchHarness) -> FastAPI:
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

    @app.post("/collect_outputs", response_model=WebSearchOutput)
    async def collect_outputs(request: RolloutRequest) -> WebSearchOutput:
        return await harness.outputs(request.rollout_id)

    return app


async def run_web_search(request: SubmitPayload) -> WebSearchOutput:
    question = _question(request.task)
    sources: list[str] = []

    async with (
        AsyncOpenAI(
            api_key=request.ac_api_key, base_url=request.ac_api_url
        ) as openai_client,
        httpx.AsyncClient(
            base_url="https://en.wikipedia.org",
            headers={"User-Agent": "ac2-cookbook/1.0"},
            timeout=20,
        ) as search_client,
    ):

        @function_tool
        async def search_wikipedia(query: str) -> str:
            """Search Wikipedia for pages relevant to a factual question."""
            response = await search_client.get(
                "/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                },
            )
            response.raise_for_status()
            results = _WikipediaResponse.model_validate_json(
                response.content
            ).query.search[:5]
            rows = []
            for result in results:
                url = f"https://en.wikipedia.org/wiki/{quote(result.title.replace(' ', '_'))}"
                sources.append(url)
                snippet = html.unescape(_HTML_TAG.sub("", result.snippet))
                rows.append({"title": result.title, "snippet": snippet, "url": url})
            return json.dumps(rows)

        agent = Agent(
            name="Web research agent",
            instructions=(
                "Answer the user's factual question. Search Wikipedia before answering, "
                "then give a concise answer based on the search results."
            ),
            model=OpenAIResponsesModel(model="policy", openai_client=openai_client),
            tools=[search_wikipedia],
        )
        result = await Runner.run(
            agent,
            input=question,
            max_turns=8,
            run_config=RunConfig(tracing_disabled=True),
        )
    return WebSearchOutput(
        answer=str(result.final_output), sources=list(dict.fromkeys(sources))
    )


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
    uvicorn.run(create_app(WebSearchHarness()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
