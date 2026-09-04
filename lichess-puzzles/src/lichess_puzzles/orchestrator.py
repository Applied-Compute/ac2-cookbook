"""Multi-turn control flow for one puzzle line."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar

from ac2.runtime import Agent, EnvironmentProtocol, Episode, Input, Message, OrchestratorProtocol
from ac2.runtime.orchestration.streams.types import StreamEvent

from .agent import (
    LichessPuzzleAgent,
    LichessPuzzleQwen36Agent,
)
from .environment import LichessPuzzleEnvironment


class LichessPuzzleOrchestrator(OrchestratorProtocol):
    """Ask for one player move, auto-play the reply, and repeat until termination."""

    agent_type: ClassVar[type[Agent]] = LichessPuzzleAgent
    environment_type: ClassVar[type[EnvironmentProtocol]] = LichessPuzzleEnvironment

    def __init__(self) -> None:
        self.env = self.environment_type()
        self._agent = self.agent_type()
        self.agents = [self._agent]
        self._episode: Episode | None = None

    async def run(self, turn_input: Input) -> AsyncIterator[StreamEvent]:
        if self._episode is None:
            self._episode = self.session.start_episode(self._agent)
            turn_input = [Message(role="user", content=self.env.initial_position_prompt())]

        async for event in self.session.run_turn(self._episode, turn_input):
            yield event

        if self.env.outcome == "active":
            self.env.mark_incomplete("agent turn ended before the puzzle terminated")


class LichessPuzzleQwen36Orchestrator(LichessPuzzleOrchestrator):
    """Static managed-project orchestrator for the Qwen3.6 baseline."""

    agent_type = LichessPuzzleQwen36Agent
