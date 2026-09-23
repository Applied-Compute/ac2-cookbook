import asyncio
import json
from functools import partial

import httpx
import pytest
from ac2.runtime import BlackBoxRolloutStatus, Message

from byoh_geo3k_vlm.orchestrator import Geo3KByohOrchestrator


@pytest.fixture
def message() -> Message:
    return Message(
        role="user",
        content=[
            {"type": "text", "text": "Find x."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,diagram"}},
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect_count", [0, 1, 3])
async def test_redirects_collect_the_same_generation(
    monkeypatch: pytest.MonkeyPatch, message: Message, redirect_count: int
) -> None:
    requests: list[httpx.Request] = []

    def relay(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        hop = int(request.url.params.get("__modal_attempt_token", "0"))
        if hop < redirect_count:
            return httpx.Response(303, headers={"location": f"/v1/chat/completions?__modal_attempt_token={hop + 1}"})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-geometry",
                "object": "chat.completion",
                "created": 0,
                "model": "model-selected-by-ac2-job",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "3"}}],
            },
        )

    monkeypatch.setattr(
        "byoh_geo3k_vlm.orchestrator.httpx.AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(relay)),
    )
    orchestrator = Geo3KByohOrchestrator()
    rollout_id = await orchestrator.submit("https://relay.example/v1", "test-token", [message])
    assert await orchestrator.get_status(rollout_id) == BlackBoxRolloutStatus.RUNNING
    async with asyncio.timeout(1):
        while await orchestrator.get_status(rollout_id) == BlackBoxRolloutStatus.RUNNING:
            await asyncio.sleep(0)

    assert await orchestrator.get_status(rollout_id) == BlackBoxRolloutStatus.COMPLETED
    assert await orchestrator.collect_outputs(rollout_id) == {"answer": "3", "finish_reason": "stop"}
    assert [request.method for request in requests] == ["POST"] + ["GET"] * redirect_count
    assert all(request.headers["authorization"] == "Bearer test-token" for request in requests)
    assert json.loads(requests[0].content)["messages"][1] == message.model_dump(include={"role", "content"})
    assert all(request.content == b"" for request in requests[1:])


@pytest.mark.asyncio
async def test_backend_failure_is_reported_without_resubmission(
    monkeypatch: pytest.MonkeyPatch, message: Message
) -> None:
    requests: list[httpx.Request] = []

    def relay(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(502, json={"detail": "Completion backend failed"})

    monkeypatch.setattr(
        "byoh_geo3k_vlm.orchestrator.httpx.AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(relay)),
    )
    orchestrator = Geo3KByohOrchestrator()
    rollout_id = await orchestrator.submit("https://relay.example/v1", "test-token", [message])
    async with asyncio.timeout(1):
        while await orchestrator.get_status(rollout_id) == BlackBoxRolloutStatus.RUNNING:
            await asyncio.sleep(0)
    assert await orchestrator.get_status(rollout_id) == BlackBoxRolloutStatus.ERRORED
    with pytest.raises(httpx.HTTPStatusError, match="502"):
        await orchestrator.collect_outputs(rollout_id)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["timeout", "teardown"])
async def test_pending_http_request_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, message: Message, finish: str
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def relay(request: httpx.Request) -> httpx.Response:
        started.set()
        try:
            await asyncio.Future[None]()
        finally:
            cancelled.set()
        raise AssertionError("The pending request should be cancelled")

    monkeypatch.setattr(
        "byoh_geo3k_vlm.orchestrator.httpx.AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(relay)),
    )
    monkeypatch.setattr("byoh_geo3k_vlm.orchestrator.COMPLETION_TIMEOUT_S", 0.05)
    orchestrator = Geo3KByohOrchestrator()
    rollout_id = await orchestrator.submit("https://relay.example/v1", "test-token", [message])
    async with asyncio.timeout(1):
        await started.wait()
        if finish == "teardown":
            await orchestrator.teardown()
        else:
            while await orchestrator.get_status(rollout_id) == BlackBoxRolloutStatus.RUNNING:
                await asyncio.sleep(0)
            assert await orchestrator.get_status(rollout_id) == BlackBoxRolloutStatus.ERRORED
            with pytest.raises(TimeoutError):
                await orchestrator.collect_outputs(rollout_id)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_rejects_continuation(message: Message) -> None:
    with pytest.raises(ValueError, match="one user turn"):
        await Geo3KByohOrchestrator().submit("https://relay.example/v1", "test-token", [message], "previous")


@pytest.mark.asyncio
async def test_rejects_task_without_image() -> None:
    message = Message(role="user", content=[{"type": "text", "text": "Find x."}])
    with pytest.raises(ValueError, match="require an image"):
        await Geo3KByohOrchestrator().submit("https://relay.example/v1", "test-token", [message])
