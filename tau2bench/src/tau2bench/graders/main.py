"""Grader for tau2bench.

Five reward components, multiplicatively composed via ``reward_basis``
(default ``[DB, COMMUNICATE]``):

- ``ACTION``      — every expected tool call was made (binary).
- ``COMMUNICATE`` — every required substring appears in an assistant message
  (case-insensitive, commas stripped).
- ``DB``          — agent (and, for telecom, user) DB hash matches a fresh
  gold env that replayed the expected actions on the same initial state.
- ``ENV_ASSERTION`` — every ``assert_*`` env method returns its expected value.
- ``NL_ASSERTION``  — every natural-language assertion is judged ``true`` by
  an LLM judge.

Premature termination (``env.terminated`` set by anything other than the
``transfer_to_human_agents`` path) short-circuits the score to 0.
"""

import hashlib
import inspect
import json
from enum import Enum
from typing import Any

from ac2.runtime import (
    EnvironmentProtocol,
    FunctionCall,
    GraderOutput,
    LLMGrader,
    Message,
    ModelConfiguration,
    Trace,
)


class RewardType(str, Enum):
    DB = "DB"
    ENV_ASSERTION = "ENV_ASSERTION"
    ACTION = "ACTION"
    COMMUNICATE = "COMMUNICATE"
    NL_ASSERTION = "NL_ASSERTION"


DEFAULT_REWARD_BASIS = [RewardType.DB.value, RewardType.COMMUNICATE.value]

NL_ASSERTIONS_SYSTEM_PROMPT = """\
TASK
- You will be given a list of expected outcomes and a conversation that was collected during a test case run.
- The conversation is between an agent and a customer.
- Your job is to evaluate whether the agent satisfies each of the expected outcomes.
- Grade each expected outcome individually.

FORMAT
- Your response should be a JSON object with the following fields:
- `reasoning`: a short explanation for your classification
- `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
- `expectedOutcome`: repeat the expectation from the input that you are grading

Example response structure:
{
    "results": [
        {
            "expectedOutcome": "<one of the expected outcomes from the input>",
            "reasoning": "<reasoning trace>",
            "metExpectation": <false or true>,
        }
    ]
}
"""


# ── Trace helpers ────────────────────────────────────────────────────────


def _all_items(trace: Trace) -> list[Any]:
    items: list[Any] = []
    for episode in trace:
        items.extend(episode.get_items())
    return items


def _extract_tool_calls(trace: Trace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _all_items(trace):
        if isinstance(item, FunctionCall):
            try:
                args = json.loads(item.arguments) if item.arguments else {}
            except json.JSONDecodeError:
                args = {}
            out.append({"name": item.name, "arguments": args})
    return out


def _extract_assistant_messages(trace: Trace) -> list[str]:
    out: list[str] = []
    for item in _all_items(trace):
        if (
            isinstance(item, Message)
            and item.role == "assistant"
            and isinstance(item.content, str)
            and item.content.strip()
        ):
            out.append(item.content)
    return out


def _conversation_for_judge(trace: Trace) -> str:
    lines = []
    for item in _all_items(trace):
        if isinstance(item, Message) and isinstance(item.content, str) and item.content:
            lines.append(f"{item.role}: {item.content}")
    return "\n".join(lines)


# ── Action evaluation ────────────────────────────────────────────────────


def _compare_action(
    expected_action: dict[str, Any],
    tool_call: dict[str, Any],
    domain: str | None,
) -> bool:
    expected_name = expected_action.get("name", "")
    actual_name = tool_call.get("name", "")
    if domain and actual_name.startswith(f"{domain}_"):
        actual_name = actual_name[len(f"{domain}_") :]
    if expected_name != actual_name:
        return False

    expected_args = expected_action.get("arguments", {})
    actual_args = tool_call.get("arguments", {})
    compare_args = expected_action.get("compare_args")
    keys = compare_args if compare_args is not None else list(actual_args.keys())
    if not keys:
        return True
    actual_filtered = {k: v for k, v in actual_args.items() if k in keys}
    expected_filtered = {k: v for k, v in expected_args.items() if k in keys}
    return actual_filtered == expected_filtered


def _evaluate_actions(
    expected_actions: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    domain: str | None,
) -> dict[str, Any]:
    if not expected_actions:
        return {"score": 1.0, "checks": [], "total_expected": 0, "total_matched": 0}
    checks = []
    for expected in expected_actions:
        match = next((tc for tc in tool_calls if _compare_action(expected, tc, domain)), None)
        checks.append({"action": expected, "matched": match is not None, "matched_call": match})
    score = 1.0 if all(c["matched"] for c in checks) else 0.0
    return {
        "score": score,
        "checks": checks,
        "total_expected": len(expected_actions),
        "total_matched": sum(1 for c in checks if c["matched"]),
    }


# ── Communicate evaluation ───────────────────────────────────────────────


def _evaluate_communicate(communicate_info: list[str], assistant_messages: list[str]) -> dict[str, Any]:
    if not communicate_info:
        return {"score": 1.0, "checks": [], "total_expected": 0, "total_matched": 0}
    checks = []
    for info in communicate_info:
        target = info.lower()
        matched = any(target in m.lower().replace(",", "") for m in assistant_messages)
        checks.append({"info": info, "met": matched})
    score = 1.0 if all(c["met"] for c in checks) else 0.0
    return {
        "score": score,
        "checks": checks,
        "total_expected": len(communicate_info),
        "total_matched": sum(1 for c in checks if c["met"]),
    }


# ── DB evaluation ────────────────────────────────────────────────────────


def _hash_dict(obj: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _replay_expected_actions(env: EnvironmentProtocol, expected_actions: list[dict[str, Any]]) -> None:
    for action in expected_actions:
        name = action.get("name", "")
        if env.domain and name.startswith(f"{env.domain}_"):  # type: ignore[attr-defined]
            name = name[len(f"{env.domain}_") :]  # type: ignore[attr-defined]
        method = getattr(env, name, None)
        if method is None:
            continue
        try:
            await _maybe_await(method(**action.get("arguments", {})))
        except Exception:
            pass


def _agent_db_hash(env: EnvironmentProtocol) -> str | None:
    db_state = getattr(env, "db_state", None)
    if db_state is None:
        return None
    state = db_state()
    return _hash_dict(state) if state else None


def _user_db_hash(env: EnvironmentProtocol) -> str | None:
    user_db_state = getattr(env, "user_db_state", None)
    if user_db_state is None:
        return None
    state = user_db_state()
    return _hash_dict(state) if state else None


async def _evaluate_db(env: EnvironmentProtocol, grader_params: dict[str, Any]) -> dict[str, Any]:
    expected_actions = grader_params.get("expected_actions", []) or []
    initial_state = grader_params.get("initial_state")

    predicted_agent = _agent_db_hash(env)
    predicted_user = _user_db_hash(env)

    gold_env = type(env)()
    gold_agent: str | None = None
    gold_user: str | None = None
    try:
        await gold_env.setup({"initial_state": initial_state} if initial_state else {})
        await _replay_expected_actions(gold_env, expected_actions)
        gold_agent = _agent_db_hash(gold_env)
        gold_user = _user_db_hash(gold_env)
    finally:
        await gold_env.teardown()

    agent_match = predicted_agent is not None and predicted_agent == gold_agent
    user_match = predicted_user == gold_user  # both None counts as a match
    db_match = agent_match and user_match
    return {
        "score": 1.0 if db_match else 0.0,
        "agent_match": agent_match,
        "user_match": user_match,
        "predicted_agent_hash": predicted_agent,
        "predicted_user_hash": predicted_user,
        "gold_agent_hash": gold_agent,
        "gold_user_hash": gold_user,
    }


# ── Env assertion evaluation ─────────────────────────────────────────────


async def _evaluate_env_assertions(env: EnvironmentProtocol, assertions: list[dict[str, Any]]) -> dict[str, Any]:
    if not assertions:
        return {"score": 1.0, "checks": [], "total_expected": 0, "total_passed": 0}

    checks = []
    score = 1.0
    domain = getattr(env, "domain", None)
    for assertion in assertions:
        env_type = assertion.get("env_type", "assistant")
        func_name = assertion.get("func_name", "")
        args = assertion.get("arguments", {}) or {}
        expected = assertion.get("assert_value", True)

        if env_type == "user" and domain not in ("telecom", "banking"):
            checks.append({"assertion": assertion, "met": True, "skipped": True})
            continue

        method = getattr(env, func_name, None)
        if method is None:
            checks.append({"assertion": assertion, "met": False, "error": "missing"})
            score = 0.0
            continue
        try:
            result = await _maybe_await(method(**args))
            met = bool(result) == bool(expected)
        except Exception as e:
            met = False
            checks.append({"assertion": assertion, "met": False, "error": str(e)})
            score = 0.0
            continue
        checks.append({"assertion": assertion, "met": met})
        if not met:
            score = 0.0

    total_passed = sum(1 for c in checks if c["met"])
    return {
        "score": score,
        "checks": checks,
        "total_expected": len(assertions),
        "total_passed": total_passed,
    }


# ── Composed reward ──────────────────────────────────────────────────────


def _normalize_reward_basis(grader_params: dict[str, Any]) -> list[str]:
    basis = grader_params.get("reward_basis", DEFAULT_REWARD_BASIS)
    if not isinstance(basis, list):
        return DEFAULT_REWARD_BASIS
    return [b if isinstance(b, str) else b.value for b in basis]


# ── Grader ───────────────────────────────────────────────────────────────


class Tau2BenchGrader(LLMGrader):
    """Multi-component tau2bench grader.

    ``grader_params`` carries the per-task expectations:

    - ``expected_actions``  : list[{"name", "arguments", "compare_args"?}]
    - ``communicate_info``  : list[str]
    - ``initial_state``     : optional dict with ``initialization_actions``
    - ``env_assertions``    : list[{"env_type", "func_name", "arguments", "assert_value"}]
    - ``nl_assertions``     : list[str]
    - ``reward_basis``      : optional list of ``RewardType`` values
                              (default ``[DB, COMMUNICATE]``)
    """

    model_config = ModelConfiguration(model="gpt-5-mini")

    async def _grade(
        self,
        grader_params: dict | None,
        trace: Trace,
        env: EnvironmentProtocol,
    ) -> GraderOutput:
        params = grader_params or {}

        if getattr(env, "terminated", False) and not _was_transfer_to_human(trace):
            termination_reason = getattr(env, "termination_reason", None)
            reasoning = "Episode terminated abnormally before completion."
            if termination_reason:
                reasoning = f"{reasoning} termination_reason={termination_reason}"
            return GraderOutput(
                score=0.0,
                reasoning=reasoning,
                artifacts={"termination_reason": termination_reason} if termination_reason else {},
            )

        domain = params.get("domain") or getattr(env, "domain", None)
        tool_calls = _extract_tool_calls(trace)
        assistant_messages = _extract_assistant_messages(trace)
        basis = _normalize_reward_basis(params)

        action = _evaluate_actions(params.get("expected_actions", []) or [], tool_calls, domain)
        comm = _evaluate_communicate(params.get("communicate_info", []) or [], assistant_messages)

        db = await _evaluate_db(env, params) if RewardType.DB.value in basis else None
        env_assert = (
            await _evaluate_env_assertions(env, params.get("env_assertions", []) or [])
            if RewardType.ENV_ASSERTION.value in basis
            else None
        )
        nl = (
            await self._evaluate_nl(trace, params.get("nl_assertions", []) or [])
            if RewardType.NL_ASSERTION.value in basis
            else None
        )

        breakdown: dict[str, float] = {}
        score = 1.0
        if RewardType.ACTION.value in basis:
            breakdown[RewardType.ACTION.value] = action["score"]
            score *= action["score"]
        if RewardType.COMMUNICATE.value in basis:
            breakdown[RewardType.COMMUNICATE.value] = comm["score"]
            score *= comm["score"]
        if db is not None:
            breakdown[RewardType.DB.value] = db["score"]
            score *= db["score"]
        if env_assert is not None:
            breakdown[RewardType.ENV_ASSERTION.value] = env_assert["score"]
            score *= env_assert["score"]
        if nl is not None:
            breakdown[RewardType.NL_ASSERTION.value] = nl["score"]
            score *= nl["score"]

        artifacts: dict[str, Any] = {
            "reward_basis": basis,
            "reward_breakdown": breakdown,
            "action_result": action,
            "comm_result": comm,
        }
        if db is not None:
            artifacts["db_result"] = db
        if env_assert is not None:
            artifacts["env_assertion_result"] = env_assert
        if nl is not None:
            artifacts["nl_result"] = nl

        reasoning = (
            f"score={score} basis={basis} breakdown={breakdown} "
            f"actions={action['total_matched']}/{action['total_expected']} "
            f"communicate={comm['total_matched']}/{comm['total_expected']}"
        )
        return GraderOutput(score=score, reasoning=reasoning, artifacts=artifacts)

    async def _evaluate_nl(self, trace: Trace, assertions: list[str]) -> dict[str, Any]:
        if not assertions:
            return {"score": 1.0, "checks": [], "total_expected": 0, "total_met": 0}

        user_prompt = f"conversation:\n{_conversation_for_judge(trace)}\n\nexpectedOutcomes:\n{assertions}\n"
        items = [
            Message(role="system", content=NL_ASSERTIONS_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]
        try:
            result = await self.get_completion(items)
            text = ""
            for it in result.items:
                if isinstance(it, Message) and isinstance(it.content, str):
                    text = it.content
                    break
            data = _parse_judge_json(text)
            results = data.get("results", [])
        except Exception as e:
            return {
                "score": 0.0,
                "checks": [],
                "error": str(e),
                "total_expected": len(assertions),
                "total_met": 0,
            }

        checks = [
            {
                "assertion": r.get("expectedOutcome", ""),
                "met": bool(r.get("metExpectation", False)),
                "reasoning": r.get("reasoning", ""),
            }
            for r in results
        ]
        score = 1.0 if checks and all(c["met"] for c in checks) else 0.0
        return {
            "score": score,
            "checks": checks,
            "total_expected": len(assertions),
            "total_met": sum(1 for c in checks if c["met"]),
        }


def _was_transfer_to_human(trace: Trace) -> bool:
    for item in _all_items(trace):
        if isinstance(item, FunctionCall) and item.name == "transfer_to_human_agents":
            return True
    return False


def _parse_judge_json(text: str) -> dict[str, Any]:
    """Parse a JSON judge response, tolerating Markdown code fences.

    ``gpt-5-*`` and several other current judge models like to wrap JSON output
    in ```json ... ``` fences even when the system prompt asks for raw JSON,
    so a bare ``json.loads`` raises ``JSONDecodeError`` and the NL-assertion
    component falls through to a 0 score. Stripping the fence is enough.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return json.loads(cleaned)
