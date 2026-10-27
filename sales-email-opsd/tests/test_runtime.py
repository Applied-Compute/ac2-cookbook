import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from ac2.retry import RetryConfig
from ac2.runtime import CompletionStream, Message, run_eval, wire_orchestrator
from ac2.runtime.completions.types import CompletionResult

from sales_email_opsd.agent import SalesEmailAgent
from sales_email_opsd.data import Conversation
from sales_email_opsd.dataset import conversation_task
from sales_email_opsd.grader import EmDashGrader
from sales_email_opsd.user import ReplaySalesEmailUser


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("Subject: Quick chat\nCould we discuss your onboarding next week?", 1.0),
        ("Subject: Quick chat\nWe can help—could we chat next week?", 0.0),
        ("", 0.0),
    ],
)
def test_ac2_rollout_replays_all_user_turns_and_grades(
    conversation: Conversation,
    monkeypatch: pytest.MonkeyPatch,
    reply: str,
    expected: float,
) -> None:
    async def completion(**kwargs) -> CompletionStream:
        async def stream() -> AsyncIterator[CompletionResult]:
            yield CompletionResult(items=[Message(role="assistant", content=reply)])

        return CompletionStream(stream())

    model = AsyncMock(side_effect=completion)
    monkeypatch.setattr(SalesEmailAgent, "get_completion", model)
    results = asyncio.run(
        run_eval(
            lambda: wire_orchestrator(
                orchestrator=None, agent="SalesEmailAgent", env="SalesEmailEnvironment"
            ),
            EmDashGrader(),
            tasks_by_dataset={"fixture": [conversation_task(conversation)]},
            user_factory=ReplaySalesEmailUser,
            retry_config=RetryConfig(max_retries=0),
            max_parallel=1,
        )
    )
    assert results[0].error is None
    assert results[0].grades[0].score == expected
    assert model.await_count == len(conversation.messages) // 2
