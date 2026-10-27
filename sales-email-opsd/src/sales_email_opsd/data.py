"""Validated synthetic conversations bundled with the example."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    seller_name: str = Field(min_length=1)
    seller_role: str = Field(min_length=1)
    seller_company: str = Field(min_length=1)
    product: str = Field(min_length=1)
    prospect_name: str = Field(min_length=1)
    prospect_title: str = Field(min_length=1)
    prospect_company: str = Field(min_length=1)
    prospect_industry: str = Field(min_length=1)
    trigger: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    tone: str = Field(min_length=1)
    iteration_moves: list[str] = Field(min_length=1)
    n_user_turns: int = Field(ge=1, le=10)

    def render(self) -> str:
        return "\n".join(
            f"- {key}: {value}" for key, value in self.model_dump(exclude={"n_user_turns"}).items()
        )


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: str = Field(min_length=1)
    split: Literal["train", "eval"]
    persona: Persona
    messages: list[ChatMessage] = Field(min_length=2)

    @model_validator(mode="after")
    def check_turns(self) -> Self:
        if len(self.messages) % 2:
            raise ValueError("A conversation must end with an assistant reply")
        for index, message in enumerate(self.messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError(f"Message {index} must have role {expected}")
        return self


def load_conversations(path: Path) -> list[Conversation]:
    records = []
    ids: set[str] = set()
    openings: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = Conversation.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if record.conversation_id in ids:
            raise ValueError(f"Duplicate conversation ID: {record.conversation_id}")
        opening = " ".join(record.messages[0].content.casefold().split())
        if opening in openings and openings[opening] != record.split:
            raise ValueError(f"Opening appears in both train and eval: {record.conversation_id}")
        ids.add(record.conversation_id)
        openings[opening] = record.split
        records.append(record)
    if not records:
        raise ValueError(f"No conversations in {path}")
    return sorted(records, key=lambda record: record.conversation_id)
