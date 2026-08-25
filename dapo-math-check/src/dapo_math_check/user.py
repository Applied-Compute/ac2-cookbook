from __future__ import annotations

from ac2.runtime import Input, User
from ac2.runtime.base.environment import EnvironmentProtocol
from ac2.runtime.base.types import Task, Trace


class DapoMathCheckUser(User):
    _task_input: Input = []
    _delivered = False

    async def setup(self, task: Task) -> None:
        self._task_input = [item.model_copy(deep=True) for item in task.input]
        self._delivered = False

    async def respond(self, env: EnvironmentProtocol, trace: Trace) -> Input | None:
        if self._delivered:
            return None
        self._delivered = True
        return [item.model_copy(deep=True) for item in self._task_input]
