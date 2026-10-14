import json

from ac2.runtime import Message

from data_pipelines.hourly import select_failures
from data_pipelines.monitor import ScoredConversation
from data_pipelines.traces import read_conversation


def _trace_line(*, truncated: bool = False) -> str:
    return json.dumps(
        {
            "trace_id": "trace_123",
            "spans": [
                {
                    "status": "completed",
                    "input": json.dumps([{"type": "message", "role": "user", "content": "Return JSON"}]),
                    "output": json.dumps([{"type": "message", "role": "assistant", "content": '{"answer":"ok","confidence":1}'}]),
                    "attributes": json.dumps({"payload_truncated": truncated}),
                }
            ],
        }
    )


def test_trace_reader_keeps_complete_text_conversation() -> None:
    conversation = read_conversation(line=_trace_line())

    assert conversation is not None
    assert conversation.trace_id == "trace_123"
    assert conversation.prompt == [Message(role="user", content="Return JSON")]
    assert conversation.answer == '{"answer":"ok","confidence":1}'


def test_trace_reader_skips_truncated_conversation() -> None:
    assert read_conversation(line=_trace_line(truncated=True)) is None


def test_failure_filter_builds_training_prompt_with_trace_provenance() -> None:
    conversation = read_conversation(line=_trace_line())
    assert conversation is not None
    results = [
        ScoredConversation(conversation=conversation, score=0, reasoning="invalid format"),
        ScoredConversation(conversation=conversation, score=1, reasoning="valid format"),
    ]

    tasks = select_failures.fn(results=results)

    assert len(tasks) == 1
    assert tasks[0].input == [Message(role="user", content="Return JSON")]
    assert tasks[0].grader_params == {"required_format": "answer-confidence-json"}
    assert tasks[0].tags == {"source_trace_id": "trace_123", "selection_reason": "invalid format"}
