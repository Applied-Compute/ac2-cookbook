"""LLM-driven customer simulator for tau2bench.

The harness calls ``respond`` with the agent's latest reply and gets back the
customer's next message, or ``None`` when the customer ends the conversation.

The simulator keeps its own view of the conversation in *flipped* role form:
its own messages are ``assistant``, the agent's text replies are ``user``, and
tool-call traffic is hidden entirely.
"""

from __future__ import annotations

from openai import AsyncOpenAI

STOP_SIGNALS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")

USER_SIMULATOR_MODEL = "gpt-5.6-luna"

GLOBAL_GUIDELINES = """\
# User Simulation Guidelines
You are playing the role of a customer contacting a customer service representative.
Your goal is to simulate realistic customer interactions while following specific scenario instructions.

## Core Principles
- Generate one message at a time, maintaining natural conversation flow.
- Strictly follow the scenario instructions you have received.
- Never make up or hallucinate information not provided in the scenario instructions. Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language to convey the same information.
- Disclose information progressively. Wait for the agent to ask for specific information before providing it.

## Task Completion
- The goal is to continue the conversation until the task is complete.
- If the instruction goal is satisfied, generate the '###STOP###' token to end the conversation.
- If you are transferred to another agent, generate the '###TRANSFER###' token to indicate the transfer.
- If you find yourself in a situation in which the scenario does not provide enough information for you to continue the conversation, generate the '###OUT-OF-SCOPE###' token to end the conversation.

Remember: The goal is to create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency.
"""

INITIAL_AGENT_GREETING = "Hi! How can I help you today?"


class Tau2BenchUser:
    def __init__(self, instructions: str, client: AsyncOpenAI) -> None:
        self._client = client
        self._messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": f"{GLOBAL_GUIDELINES}\n<scenario>\n{instructions}\n</scenario>",
            }
        ]

    async def respond(self, agent_text: str | None) -> str | None:
        self._messages.append({"role": "user", "content": agent_text or INITIAL_AGENT_GREETING})
        completion = await self._client.chat.completions.create(
            model=USER_SIMULATOR_MODEL,
            messages=self._messages,
        )
        content = completion.choices[0].message.content
        if not content or any(signal in content for signal in STOP_SIGNALS):
            return None
        self._messages.append({"role": "assistant", "content": content})
        return content
