"""Online OPSD hints: a judge picks the first offending reply and writes a hint for it."""

from __future__ import annotations

import json

from ac2.runtime import (
    FunctionCall,
    GraderOutput,
    HintMutation,
    Message,
    ModelConfiguration,
    OnlineHinter,
    OnlineHintSelection,
    ResponsesAPIOptions,
    Task,
    Tool,
    Trace,
)

from .agent import SalesEmailAgent
from .prompts import JUDGE_PROMPT
from .text import message_text

JUDGE_MODEL = "gpt-5-mini"
MAX_MESSAGE_CHARS = 2500

EMIT_HINT_TOOL = Tool(
    name="emit_online_hint_selection",
    description="Select one assistant turn and the teacher-only hint to inject before it.",
    parameters={
        "type": "object",
        "properties": {
            "assistant_turn": {
                "type": "integer",
                "description": "Zero-based assistant turn number shown in the transcript.",
            },
            "hint": {
                "type": "string",
                "description": "Teacher-only hint to inject before this assistant turn.",
            },
            "reason": {"type": "string", "description": "Brief reason this turn was selected."},
        },
        "required": ["assistant_turn", "hint", "reason"],
    },
)


def render_transcript(messages: list[tuple[int, Message]]) -> str:
    lines = []
    assistant_turn = 0
    for index, message in messages:
        if message.role == "assistant":
            lines.append(f"\n[assistant_turn={assistant_turn} message_index={index}]")
            assistant_turn += 1
        else:
            lines.append(f"\n[{message.role} message_index={index}]")
        lines.append(message_text(message)[:MAX_MESSAGE_CHARS])
    return "\n".join(lines)


class SalesEmailJudgeHinter(OnlineHinter):
    def __init__(self, model: str = JUDGE_MODEL) -> None:
        self.model_configuration = ModelConfiguration(model=model, kwargs={"max_output_tokens": 2048})

    async def _select_hints(
        self, task: Task, trace: Trace, grader_output: GraderOutput
    ) -> list[OnlineHintSelection]:
        episode = next(episode for episode in trace if episode.agent_name == SalesEmailAgent.__name__)
        items = episode.get_items()
        messages = [(i, item) for i, item in enumerate(items) if isinstance(item, Message)]
        assistant_indices = [i for i, message in messages if message.role == "assistant"]
        prompt = JUDGE_PROMPT.format(
            transcript=render_transcript(messages),
            reward=grader_output.score,
            grader_reasoning=grader_output.reasoning[:2000],
            n_assistant_turns=len(assistant_indices),
        )
        result = await self.completion_client.complete(
            [Message(role="user", content=prompt)],
            tools=[EMIT_HINT_TOOL],
            stream=False,
            responses_api_options=ResponsesAPIOptions(tool_choice="required"),
        )
        call = next(
            (
                item
                for item in result.items
                if isinstance(item, FunctionCall) and item.name == EMIT_HINT_TOOL.name
            ),
            None,
        )
        if call is None:
            raise ValueError(f"Judge did not call {EMIT_HINT_TOOL.name}")
        selection = json.loads(call.arguments)
        turn = selection["assistant_turn"]
        if not 0 <= turn < len(assistant_indices):
            raise ValueError(
                f"Judge selected assistant_turn={turn}; valid turns are 0..{len(assistant_indices) - 1}"
            )
        boundary = assistant_indices[turn]
        return [
            OnlineHintSelection(
                hint=selection["hint"],
                episode_id=episode.id,
                rollout_prefix_len=boundary,
                mutation=HintMutation(item_boundary=boundary, action="insert_message", role="tool"),
            )
        ]
