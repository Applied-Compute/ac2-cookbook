import asyncio
from unittest.mock import AsyncMock

import pytest
from ac2.runtime import Episode, Message
from ac2.runtime.completions.types import CompletionResult

from sales_email_opsd.data import Conversation
from sales_email_opsd.dataset import conversation_task
from sales_email_opsd.environment import SalesEmailEnvironment
from sales_email_opsd.user import ReplaySalesEmailUser, SalesEmailUser


def test_replay_is_fixed_and_resets_between_tasks(conversation: Conversation) -> None:
    async def run():
        user = ReplaySalesEmailUser()
        task = conversation_task(conversation)
        for _ in range(2):
            await user.setup(task)
            turns = []
            while (response := await user.respond(SalesEmailEnvironment(), [])) is not None:
                turns.append(response[0].content)
                response[0].content = "Mutating a returned message cannot change replay state"
            assert turns == [m.content for m in conversation.messages if m.role == "user"]

    asyncio.run(run())


def simulated_response(conversation: Conversation, monkeypatch: pytest.MonkeyPatch, reply: str | Exception):
    result = (
        CompletionResult(items=[Message(role="assistant", content=reply)]) if isinstance(reply, str) else None
    )
    complete = AsyncMock(return_value=result, side_effect=reply if isinstance(reply, Exception) else None)
    monkeypatch.setattr("sales_email_opsd.user.CompletionClient.complete", complete)

    async def run():
        user = SalesEmailUser()
        await user.setup(conversation_task(conversation))
        await user.respond(SalesEmailEnvironment(), [])
        episode = Episode("SalesEmailAgent")
        episode.add([Message(role="assistant", content="Subject: Hello\nA draft.")])
        return await user.respond(SalesEmailEnvironment(), [episode])

    return asyncio.run(run())


def test_simulator_errors_fail_the_rollout(
    conversation: Conversation, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="provider unavailable"):
        simulated_response(conversation, monkeypatch, RuntimeError("provider unavailable"))


def test_blank_simulator_reply_is_an_error(
    conversation: Conversation, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="no visible reply"):
        simulated_response(conversation, monkeypatch, " ")


def test_exact_stop_signal_ends_conversation(
    conversation: Conversation, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert simulated_response(conversation, monkeypatch, "###DONE###") is None


def test_embedded_stop_signal_does_not_end_conversation(
    conversation: Conversation, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert simulated_response(conversation, monkeypatch, "Remove the ###DONE### placeholder") is not None
