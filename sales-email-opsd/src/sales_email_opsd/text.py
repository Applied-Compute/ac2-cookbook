"""Extract visible assistant text without counting reasoning or user input."""

import re

from ac2.runtime import Message, Trace

# Qwen3 served through AC2 returns its reasoning inline as a leading <think> block, not a Reasoning item.
THINK_BLOCK = re.compile(r"^\s*<think>.*?(?:</think>|$)\s*", re.DOTALL)


def message_text(message: Message) -> str:
    if isinstance(message.content, str):
        text = message.content
    else:
        text = "".join(
            part["text"]
            for part in message.content
            if part.get("type") in {"text", "output_text"} and isinstance(part.get("text"), str)
        )
    return THINK_BLOCK.sub("", text, count=1)


def assistant_texts(trace: Trace) -> list[str]:
    return [
        message_text(item)
        for episode in trace
        for item in episode.get_items()
        if isinstance(item, Message) and item.role == "assistant"
    ]
