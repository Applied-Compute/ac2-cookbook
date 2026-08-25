from typing import Annotated

from ac2.runtime import Environment, FunctionCall, FunctionCallOutput, Item, tool
from pydantic import Field

from .answer_utils import answers_match

MAX_INCORRECT = 3


class DapoMathCheckEnvironment(Environment):
    _expected_answer: str = ""
    _incorrect_count: int = 0
    _terminated_reason: str = ""
    _final_answer: str = ""

    async def setup(self, env_params: dict) -> None:
        self._expected_answer = str((env_params or {}).get("answer", ""))
        self._incorrect_count = 0
        self._terminated_reason = ""
        self._final_answer = ""

    async def teardown(self) -> None:
        return None

    async def inspect(self, fields: list[str]) -> dict:
        return {}

    @property
    def incorrect_count(self) -> int:
        return self._incorrect_count

    @property
    def terminated_reason(self) -> str:
        return self._terminated_reason

    @property
    def is_terminated(self) -> bool:
        return bool(self._terminated_reason)

    @property
    def terminated_correct(self) -> bool:
        if self._terminated_reason == "correct":
            return True
        if self._terminated_reason == "finished":
            return answers_match(self._final_answer, self._expected_answer)
        return False

    async def step(self, items: list[Item]) -> tuple[list[FunctionCallOutput], bool]:
        calls = [item for item in items if isinstance(item, FunctionCall)]
        if not calls:
            return [], True
        outputs: list[FunctionCallOutput] = []
        for call in calls:
            outputs.append(await self.call_tool(call))
            if self.is_terminated:
                break
        return outputs, self.is_terminated

    @tool("Verify a candidate answer. Returns 'Correct.' or 'Incorrect.' Three incorrect submissions end the task.")
    async def check_answer(
        self,
        answer: Annotated[str, Field(description="The candidate answer, such as '42' or '1/3'.")],
    ) -> str:
        if answers_match(answer, self._expected_answer):
            self._terminated_reason = "correct"
            return "Correct."
        self._incorrect_count += 1
        if self._incorrect_count >= MAX_INCORRECT:
            self._terminated_reason = "exhausted"
        return "Incorrect."

    @tool("Finish the task with your final answer. This answer is graded directly and ends the task.")
    async def finish(
        self,
        answer: Annotated[str, Field(description="Your final answer, such as '42' or '1/3'.")],
    ) -> str:
        self._final_answer = answer
        self._terminated_reason = "finished"
        return "Task finished."
