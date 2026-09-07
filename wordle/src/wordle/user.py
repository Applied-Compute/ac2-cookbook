from ac2.runtime import Input, Message, Task, Trace, User
from ac2.runtime.base.environment import EnvironmentProtocol

from .environment import WordleEnvironment


class WordleUser(User):
    """Continue premature text answers within the same, budgeted episode."""

    async def setup(self, task: Task) -> None:
        self._input = [item.model_copy(deep=True) for item in task.input]
        self._delivered = False
        self._reminders = 0

    async def respond(self, env: EnvironmentProtocol, trace: Trace) -> Input | None:
        if not isinstance(env, WordleEnvironment):
            raise TypeError("WordleUser requires WordleEnvironment.")
        if env.is_terminated:
            return None
        if not self._delivered:
            self._delivered = True
            return [item.model_copy(deep=True) for item in self._input]
        if self._reminders >= 3:
            env.terminated_reason = "abandoned"
            return None
        self._reminders += 1
        return [Message(role="user", content=(
            "The game is not over. Submit your next guess using check_answer. "
            "A written answer does not count as a guess."
        ))]
