import asyncio

import pytest
from ac2.runtime import Episode, Message, Reasoning

from sales_email_opsd.dashes import count_dashes
from sales_email_opsd.environment import SalesEmailEnvironment
from sales_email_opsd.grader import EmDashGrader


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a—b", 1),
        ("—", 1),
        ("a – b", 1),
        ("café–résumé", 1),
        ("a--b", 1),
        ("a -- b", 1),
        ("a - b", 1),
        ("a—b—c", 2),
        ("Hi - can we talk?", 1),
        ("attack-surface-management", 0),
        ("20-minute call", 0),
        ("5-10", 0),
        ("5–10", 0),
        ("5 – 10", 0),
        ("5 -- 10", 0),
        ("- bullet", 0),
        ("Header\n - bullet\n - bullet", 0),
        ("Header\n---\nBody", 0),
        ("---", 0),
        ("", 0),
        ("one: two", 0),
    ],
)
def test_dash_policy(text: str, expected: int) -> None:
    assert count_dashes(text) == expected


def grade(*messages: Message | Reasoning):
    episode = Episode("SalesEmailAgent")
    episode.add(list(messages))
    return asyncio.run(EmDashGrader()._grade(None, [episode], SalesEmailEnvironment()))


@pytest.mark.parametrize("text", ["", " "])
def test_empty_reply_scores_zero(text: str) -> None:
    assert grade(Message(role="assistant", content=text)).score == 0.0


@pytest.mark.parametrize("text", ["No thanks.", "Subject: Hello", "Subject:\nBody"])
def test_non_email_reply_is_flagged_invalid(text: str) -> None:
    assert grade(Message(role="assistant", content=text)).artifacts["valid_output"] is False


def test_absent_assistant_fails() -> None:
    assert grade(Message(role="user", content="Write an email.")).score == 0.0


def test_user_and_reasoning_dashes_are_not_scored() -> None:
    output = grade(
        Message(role="user", content="Write—please"),
        Reasoning(content="Think—then write"),
        Message(role="assistant", content="Subject: Hello\nWant a 20-minute demo?"),
    )
    assert output.score == 1.0


def test_inline_think_block_is_not_scored() -> None:
    reply = "<think>\nDraft it—then check.\n</think>\n\nSubject: Hello\nWant a 20-minute demo?"
    assert grade(Message(role="assistant", content=reply)).score == 1.0


def test_unclosed_think_block_is_not_an_email() -> None:
    assert grade(Message(role="assistant", content="<think>\nSubject: Hello\nStill thinking")).score == 0.0


def test_text_content_parts_are_scored() -> None:
    output = grade(Message(role="assistant", content=[{"type": "text", "text": "Subject: Hello\nHi—there"}]))
    assert output.artifacts["dash_count"] == 1


def test_padding_cannot_dilute_a_violation_into_a_pass() -> None:
    output = grade(Message(role="assistant", content="Subject: Hello\nHi—there " + "padding " * 10000))
    assert output.score == 0.0


def test_score_is_the_share_of_dash_free_replies() -> None:
    output = grade(
        Message(role="assistant", content="Subject: Hello\nHi—there"),
        Message(role="assistant", content="Subject: Hello\nHi there"),
    )
    assert output.score == 0.5


def test_truncated_conversation_is_not_a_strict_pass() -> None:
    episode = Episode("SalesEmailAgent")
    episode.add([Message(role="assistant", content="Subject: Hello\nWe can help with")])
    episode.truncated = True
    result = asyncio.run(EmDashGrader()._grade(None, [episode], SalesEmailEnvironment()))
    assert result.artifacts["clean"] is False
