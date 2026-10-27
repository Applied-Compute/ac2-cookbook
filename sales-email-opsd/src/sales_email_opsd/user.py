"""Fixed user-message replay for comparisons; optional adaptive user simulation."""

from __future__ import annotations

import asyncio

from ac2.runtime import (
    CompletionClient,
    EnvironmentProtocol,
    Input,
    Message,
    ModelConfiguration,
    Task,
    Trace,
    User,
)
from pydantic import BaseModel, ConfigDict, Field

from .data import Persona
from .text import assistant_texts, message_text

STOP_SIGNAL = "###DONE###"
USER_SIM_TIMEOUT_SECONDS = 180.0
GUIDELINES = """Play the sales rep described below, iterating on an outbound email.
Write one short message at a time. Give concrete feedback on content, length, or tone
from your iteration moves. Do not invent facts or comment on punctuation or typography.
When satisfied, reply with exactly ###DONE###.
"""


class UserParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona: Persona
    followups: list[str] = Field(default_factory=list)


def opening_message(task: Task) -> Message:
    if len(task.input) != 1 or not isinstance(task.input[0], Message) or task.input[0].role != "user":
        raise ValueError("Sales email tasks must start with exactly one user message")
    opening = task.input[0]
    if not message_text(opening).strip():
        raise ValueError("The opening user message must not be empty")
    return opening.model_copy(deep=True)


class ReplaySalesEmailUser(User):
    """Use identical recorded user turns for every candidate model."""

    async def setup(self, task: Task) -> None:
        params = UserParameters.model_validate(task.env_params)
        self._messages = [
            opening_message(task),
            *(Message(role="user", content=text) for text in params.followups),
        ]
        if any(not message_text(message).strip() for message in self._messages):
            raise ValueError("Replay followups must not be empty")
        self._position = 0

    async def respond(self, env: EnvironmentProtocol, trace: Trace) -> Input | None:
        if self._position == len(self._messages):
            return None
        message = self._messages[self._position].model_copy(deep=True)
        self._position += 1
        return [message]


class SalesEmailUser(User):
    """Adaptive simulator for exploratory runs; failures remain rollout errors."""

    def __init__(self, model: str = "gpt-4.1-2025-04-14") -> None:
        self.model_configuration = ModelConfiguration(model=model, kwargs={"temperature": 0.0})

    async def setup(self, task: Task) -> None:
        params = UserParameters.model_validate(task.env_params)
        self._opening = opening_message(task)
        self._persona = params.persona
        self._history: list[Message] = []
        self._turns = 0
        self._stopped = False
        self._client = CompletionClient(self.model_configuration)

    async def respond(self, env: EnvironmentProtocol, trace: Trace) -> Input | None:
        if self._stopped or self._turns >= self._persona.n_user_turns:
            return None
        if self._turns == 0:
            self._turns = 1
            self._history.append(Message(role="assistant", content=message_text(self._opening)))
            return [self._opening.model_copy(deep=True)]
        replies = assistant_texts(trace)
        if not replies or not replies[-1].strip():
            raise ValueError("Cannot simulate feedback without an assistant reply")
        self._history.append(Message(role="user", content=replies[-1]))
        result = await asyncio.wait_for(
            self._client.complete(
                items=[
                    Message(role="system", content=GUIDELINES + "\n" + self._persona.render()),
                    *self._history,
                ]
            ),
            timeout=USER_SIM_TIMEOUT_SECONDS,
        )
        content = "\n".join(
            message_text(item)
            for item in result.items
            if isinstance(item, Message) and item.role == "assistant"
        ).strip()
        if not content:
            raise ValueError("The user simulator returned no visible reply")
        if content == STOP_SIGNAL:
            self._stopped = True
            return None
        self._history.append(Message(role="assistant", content=content))
        self._turns += 1
        return [Message(role="user", content=content)]
