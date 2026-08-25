from __future__ import annotations

from ac2.runtime import Agent, ModelConfiguration

SYSTEM_PROMPT = """You are solving a math problem. Two tools are available:
- check_answer(answer): verifies a candidate answer, returning "Correct." or "Incorrect." Three incorrect submissions end the task.
- finish(answer): ends the task with your final answer, which is graded directly.

Solve the problem and submit your answer."""


class DapoMathAgent(Agent):
    description = "Math agent with check_answer and finish tools."
    model_configuration = ModelConfiguration(
        model="gpt-5-mini",
        kwargs={"reasoning": {"summary": "detailed"}},
    )
    system_prompt = SYSTEM_PROMPT
