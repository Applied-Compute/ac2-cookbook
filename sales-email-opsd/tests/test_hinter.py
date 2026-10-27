import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from ac2.runtime import Episode, FunctionCall, GraderOutput, Item, Message, OnlineHintSelection, Task
from ac2.runtime.completions.types import CompletionResult

from sales_email_opsd.hinter import SalesEmailJudgeHinter


def select_hints(judge_items: list[Item]) -> list[OnlineHintSelection]:
    episode = Episode("SalesEmailAgent")
    episode.add(
        [
            Message(role="user", content="Draft an email"),
            Message(role="assistant", content="Subject: Hi\nHello"),
            Message(role="user", content="Shorter please"),
            Message(role="assistant", content="Subject: Hi\nWe help—chat?"),
        ]
    )
    hinter = SalesEmailJudgeHinter()
    hinter._completion_client = AsyncMock()
    hinter._completion_client.complete.return_value = CompletionResult(items=judge_items)
    return asyncio.run(
        hinter.select_hints(
            Task(input=[Message(role="user", content="Draft an email")]),
            [episode],
            GraderOutput(score=0.0, reasoning="dash found"),
        )
    )


def judge_call(selection: dict[str, int | str]) -> FunctionCall:
    return FunctionCall(call_id="1", name="emit_online_hint_selection", arguments=json.dumps(selection))


def test_judge_hint_is_inserted_before_the_selected_reply() -> None:
    selection = {"assistant_turn": 1, "hint": 'Rewrite "We help—chat?" without the dash.', "reason": "dash"}

    [hint] = select_hints([judge_call(selection)])

    placement = (hint.rollout_prefix_len, hint.mutation.action, hint.mutation.role)
    assert placement == (3, "insert_message", "tool")
    assert hint.hint == selection["hint"]


def test_judge_without_a_tool_call_is_an_error() -> None:
    with pytest.raises(ValueError, match="did not call"):
        select_hints([Message(role="assistant", content="No violations.")])


def test_judge_turn_outside_the_transcript_is_an_error() -> None:
    with pytest.raises(ValueError, match="valid turns are 0..1"):
        select_hints([judge_call({"assistant_turn": 2, "hint": "Drop the dash.", "reason": "dash"})])
