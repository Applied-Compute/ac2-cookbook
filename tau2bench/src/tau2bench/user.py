"""LLM-driven customer simulator for tau2bench.

The customer side of the conversation is a ``User`` (not an ``Agent``);
``User.respond()`` is called by the orchestrator driver (eval / training
generator) before each agent turn and returns the next customer ``Input``, or
``None`` to end the rollout.

We hold a private "user-sim view" of the conversation in *flipped* role form:
from the simulator's perspective its own messages are ``assistant``, the agent's
text replies are ``user``, and tool-call traffic is hidden entirely. The agent
never sees this internal view — only the ``Message(role="user", content=...)``
items returned from ``respond()``.

"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from ac2.runtime import (
    CompletionClient,
    EnvironmentProtocol,
    Message,
    ModelConfiguration,
    Task,
    Trace,
    User,
)
from ac2.runtime.base.types import Input
from ac2.tracing import Span

logger = logging.getLogger(__name__)

STOP_SIGNALS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")

# Hard wall-clock cap on a single user-sim LLM call. Sits above the OpenAI
USER_SIMULATOR_TIMEOUT_SECONDS = float(os.environ.get("USER_SIMULATOR_TIMEOUT_SECONDS", "180"))

USER_SIMULATOR_ERROR = "user_simulator_error"

DEFAULT_MAX_TURNS = 50

_active_user_sim_calls = 0
_max_active_user_sim_calls = 0

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


class Tau2BenchUser(User):
    """LLM-driven customer simulator.

    Drives the customer side of a tau2bench conversation. The eval and training
    drivers loop ``user.respond(env, trace) → orchestrator.run(turn_input)``
    until ``respond`` returns ``None``. We return ``None`` when:

    - the LLM emits a stop signal (``###STOP###`` / ``###TRANSFER###`` /
      ``###OUT-OF-SCOPE###``);
    - the env's ``terminated`` flag is already set (the agent ended the call);
    - the user-sim LLM call fails or times out (soft-fails the rollout); or
    - we've reached ``max_turns`` user turns.
    """

    # No temperature override: this model rejects the parameter (400
    # "'temperature' is not supported with this model"), which soft-fails every
    # rollout as user_simulator_error. Watch for a trigger-happy sim: one that
    # over-emits ###STOP### right after confirming ends the rollout before the
    # agent executes the DB write, which unfairly penalizes policy-compliant
    # (confirm-first) agents.
    model_configuration = ModelConfiguration(model="gpt-5.6-luna")
    max_turns = DEFAULT_MAX_TURNS

    async def setup(self, task: Task) -> None:
        self._client = CompletionClient(self.model_configuration)
        self._sim_messages: list[Message] = []
        self._stopped = False
        self._n_user_turns = 0
        self.instructions = (task.grader_params or {}).get("user_instructions", "")

    @property
    def system_prompt(self) -> str:
        return f"{GLOBAL_GUIDELINES}\n<scenario>\n{self.instructions}\n</scenario>"

    async def respond(
        self,
        env: EnvironmentProtocol,
        trace: Trace,
    ) -> Input | None:
        if self._stopped:
            return None
        if getattr(env, "terminated", False):
            return None
        if self._n_user_turns >= self.max_turns:
            logger.info("User simulator hit max_turns=%d; ending rollout.", self.max_turns)
            return None

        agent_text = _last_agent_text(trace)
        if not self._sim_messages:
            # First turn: simulate the agent saying "Hi, how can I help?" so the
            # customer has something to react to.
            seed = agent_text or INITIAL_AGENT_GREETING
            self._sim_messages.append(Message(role="user", content=seed))
        else:
            if agent_text is None:
                return None
            last_seen = self._sim_messages[-1]
            if last_seen.role == "user" and last_seen.content == agent_text:
                return None
            self._sim_messages.append(Message(role="user", content=agent_text))

        llm_input: list = [Message(role="system", content=self.system_prompt)]
        llm_input.extend(self._sim_messages)
        prompt_chars = sum(len(str(getattr(item, "content", ""))) for item in llm_input)
        agent_chars = len(agent_text or "")
        call_started = time.monotonic()

        global _active_user_sim_calls, _max_active_user_sim_calls
        _active_user_sim_calls += 1
        _max_active_user_sim_calls = max(_max_active_user_sim_calls, _active_user_sim_calls)
        active_at_start = _active_user_sim_calls

        try:
            with Span(name="user_sim_completion", role="user") as span:
                span.resource = "tau2_user_simulator"
                span.resource_type = "user"
                span.model = self.model_configuration.model
                span.input = {
                    "message_count": len(llm_input),
                    "prompt_chars": prompt_chars,
                    "agent_chars": agent_chars,
                    "user_turn": self._n_user_turns,
                    "active_user_sim_calls": active_at_start,
                    "max_active_user_sim_calls": _max_active_user_sim_calls,
                    "timeout_seconds": USER_SIMULATOR_TIMEOUT_SECONDS,
                }
                try:
                    result = await asyncio.wait_for(
                        self._client.complete(items=llm_input),
                        timeout=USER_SIMULATOR_TIMEOUT_SECONDS,
                    )
                except Exception as exc:
                    span.output = {
                        "elapsed_seconds": round(time.monotonic() - call_started, 3),
                        "active_at_start": active_at_start,
                        "active_user_sim_calls": _active_user_sim_calls,
                        "max_active_user_sim_calls": _max_active_user_sim_calls,
                        "error_type": type(exc).__name__,
                    }
                    raise
                # Record the customer's reply text (the user message sent to the
                # agent) so the turn shows in the trace, not only metadata.
                span.output = {
                    "role": "user",
                    "content": _first_sim_reply_text(result.items),
                    "elapsed_seconds": round(time.monotonic() - call_started, 3),
                    "item_count": len(result.items),
                    "active_user_sim_calls": _active_user_sim_calls,
                }
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - call_started
            logger.warning(
                "User simulator LLM call timed out after %.1fs; ending rollout. "
                "elapsed=%.1fs active_at_start=%d active_now=%d max_active=%d "
                "messages=%d prompt_chars=%d agent_chars=%d user_turn=%d",
                USER_SIMULATOR_TIMEOUT_SECONDS,
                elapsed,
                active_at_start,
                _active_user_sim_calls,
                _max_active_user_sim_calls,
                len(llm_input),
                prompt_chars,
                agent_chars,
                self._n_user_turns,
            )
            self._mark_user_sim_failure(env)
            return None
        except Exception as exc:
            elapsed = time.monotonic() - call_started
            logger.warning(
                "User simulator LLM call failed (%s); ending rollout. "
                "elapsed=%.1fs active_at_start=%d active_now=%d max_active=%d "
                "messages=%d prompt_chars=%d agent_chars=%d user_turn=%d",
                exc,
                elapsed,
                active_at_start,
                _active_user_sim_calls,
                _max_active_user_sim_calls,
                len(llm_input),
                prompt_chars,
                agent_chars,
                self._n_user_turns,
            )
            self._mark_user_sim_failure(env)
            return None
        finally:
            _active_user_sim_calls -= 1

        content = _first_sim_reply_text(result.items)
        if content is None:
            self._stopped = True
            return None
        if any(sig in content for sig in STOP_SIGNALS):
            self._stopped = True
            return None

        self._sim_messages.append(Message(role="assistant", content=content))
        self._n_user_turns += 1
        return [Message(role="user", content=content)]

    def _mark_user_sim_failure(self, env: EnvironmentProtocol) -> None:
        """Soft-fail: end the rollout and tag the env so the grader can triage.

        ``Tau2BenchGrader`` already scores 0 when ``env.terminated`` is set
        without a ``transfer_to_human_agents`` call, so flipping the flag is
        enough to keep the partial trajectory and convert this rollout into a
        normal 0-reward sample rather than a thrown exception. The
        ``termination_reason`` tag lets eval triage tell user-sim infra
        failures apart from genuine agent failures without re-running.
        """
        self._stopped = True
        env.terminated = True  # type: ignore[attr-defined]
        env.termination_reason = USER_SIMULATOR_ERROR  # type: ignore[attr-defined]


class Tau2BenchDefaultUser(Tau2BenchUser):
    pass


def _last_agent_text(trace: Trace) -> str | None:
    """Return the most recent agent-emitted text content across all episodes."""
    for episode in reversed(list(trace)):
        for item in reversed(episode.get_items()):
            if (
                isinstance(item, Message)
                and item.role == "assistant"
                and isinstance(item.content, str)
                and item.content.strip()
            ):
                return item.content
    return None


def _first_sim_reply_text(items: list) -> str | None:
    """Return the simulator's reply text — its (role-flipped) ``assistant`` message."""
    for item in items:
        if (
            isinstance(item, Message)
            and item.role == "assistant"
            and isinstance(item.content, str)
            and item.content.strip()
        ):
            return item.content
    return None
