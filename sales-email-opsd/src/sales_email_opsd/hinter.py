"""Model-written OPSD hints: a judge writes the hint for each replay task's selected reply."""

from __future__ import annotations

from ac2.runtime import HintMutation, HintSelection, Message, ModelConfiguration, OfflineHinter, Task

from .prompts import JUDGE_PROMPT
from .text import message_text

JUDGE_MODEL = "gpt-5.6-luna"


class SalesEmailJudgeHinter(OfflineHinter):
    def __init__(self, model: str = JUDGE_MODEL) -> None:
        self.model_configuration = ModelConfiguration(model=model, kwargs={"max_output_tokens": 4096})

    async def _select_hints(self, task: Task) -> list[HintSelection]:
        boundary = len(task.input) - 1
        reply = task.input[boundary]
        if not isinstance(reply, Message) or reply.role != "assistant":
            raise ValueError("A replay task must end with the assistant reply to hint")
        result = await self.completion_client.complete(
            [Message(role="user", content=JUDGE_PROMPT.format(reply=message_text(reply)))],
            stream=False,
        )
        hint = "\n".join(
            message_text(item)
            for item in result.items
            if isinstance(item, Message) and item.role == "assistant"
        ).strip()
        if not hint:
            raise ValueError("The judge returned no hint")
        return [
            HintSelection(
                hint=hint,
                rollout_prefix_len=boundary,
                mutation=HintMutation(item_boundary=boundary, action="append_to_previous_item"),
            )
        ]
