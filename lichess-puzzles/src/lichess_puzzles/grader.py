"""Deterministic, regrade-safe prefix-progress reward."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ac2.runtime import (
    EnvironmentProtocol,
    FunctionCall,
    FunctionCallOutput,
    Grader,
    GraderOutput,
    Trace,
)

from .chess_logic import assistant_text
from .tool_contract import (
    ACTION_INTERFACE,
    DEFAULT_TRIES,
    SUBMIT_MOVE_TOOL,
    parse_submit_move_arguments,
)

REWARD_SCHEME = "normalized_prefix_plus_completion_v1"
PROGRESS_REWARD_WEIGHT = 0.5


@dataclass(frozen=True)
class ActionExtraction:
    """Regrade-safe action sequence plus interface diagnostics."""

    actions: list[str | None]
    interface: str
    malformed_reason: str = ""
    format_retry_reasons: tuple[str, ...] = ()
    wrong_move_retry_actions: tuple[str, ...] = ()
    terminal_wrong_move_actions: tuple[str, ...] = ()


def _tool_output_diagnostic(output: FunctionCallOutput) -> tuple[str, str]:
    try:
        payload = json.loads(output.output.partition("\n")[0])
    except (json.JSONDecodeError, TypeError):
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    status = str(payload.get("status", ""))
    error_type = payload.get("error_type")
    return status, str(error_type) if error_type else ""


def _parse_tool_call(call: FunctionCall) -> tuple[str | None, str]:
    if call.name != SUBMIT_MOVE_TOOL:
        return None, "wrong_tool"
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return None, "invalid_tool_json"
    return parse_submit_move_arguments(arguments)


def extract_actions(trace: Trace) -> ActionExtraction:
    """Extract structured move submissions and retry diagnostics from a trace.

    Episodes store a flat item sequence, so assistant prose and reasoning emitted
    alongside a tool call must be associated with that call rather than scored as
    an independent text submission. Text becomes an action only when a completion
    reaches its tool output (or the end of an episode) without a function call.
    """

    outputs: dict[str, FunctionCallOutput] = {}
    for episode in trace:
        for item in episode.get_items():
            if isinstance(item, FunctionCallOutput):
                outputs[item.call_id] = item

    actions: list[str | None] = []
    malformed_reason = ""
    format_retry_reasons: list[str] = []
    wrong_move_retry_actions: list[str] = []
    terminal_wrong_move_actions: list[str] = []
    def record_call(call: FunctionCall) -> None:
        nonlocal malformed_reason

        output = outputs.get(call.call_id)
        output_status, output_error = (
            _tool_output_diagnostic(output) if output is not None else ("", "")
        )
        if output_status == "format_error":
            format_retry_reasons.append(output_error or "format_error")
            return
        if output_status == "wrong_move":
            action, call_error = _parse_tool_call(call)
            if action is None:
                actions.append(None)
                malformed_reason = malformed_reason or call_error or "invalid_tool_arguments"
            else:
                wrong_move_retry_actions.append(action)
            return
        if output_status == "incorrect":
            action, call_error = _parse_tool_call(call)
            actions.append(action)
            if action is None:
                malformed_reason = malformed_reason or call_error or "invalid_tool_arguments"
            else:
                terminal_wrong_move_actions.append(action)
            return
        if output_status == "malformed":
            actions.append(None)
            malformed_reason = malformed_reason or output_error or "malformed_tool_call"
            return
        action, call_error = _parse_tool_call(call)
        actions.append(action)
        malformed_reason = malformed_reason or call_error

    for episode in trace:
        pending_completion_items: list[Any] = []
        for item in episode.get_items():
            if isinstance(item, FunctionCallOutput):
                if any(
                    (text := assistant_text([pending_item])) is not None and text.strip()
                    for pending_item in pending_completion_items
                ):
                    actions.append(None)
                    malformed_reason = malformed_reason or "text_submission"
                pending_completion_items.clear()
                continue

            if isinstance(item, FunctionCall):
                # The pending items belong to this completion, which has a
                # structured action. The environment ignores companion prose.
                pending_completion_items.clear()
                record_call(item)
                continue

            pending_completion_items.append(item)

        if any(
            (text := assistant_text([pending_item])) is not None and text.strip()
            for pending_item in pending_completion_items
        ):
            actions.append(None)
            malformed_reason = malformed_reason or "text_submission"
    return ActionExtraction(
        actions,
        ACTION_INTERFACE,
        malformed_reason,
        tuple(format_retry_reasons),
        tuple(wrong_move_retry_actions),
        tuple(terminal_wrong_move_actions),
    )


def grade_actions(expected: list[str], actual: list[str | None]) -> tuple[str, int, str]:
    """Return outcome, correct prefix length, and a concise explanation."""

    correct = 0
    for index, expected_move in enumerate(expected):
        if index >= len(actual):
            return "incomplete", correct, f"missing player move {index + 1} of {len(expected)}"
        action = actual[index]
        if action is None:
            return "malformed", correct, f"player submission {index + 1} was malformed"
        if action != expected_move:
            return (
                "incorrect",
                correct,
                f"move {index + 1}: expected {expected_move}, received {action}",
            )
        correct += 1
    if len(actual) > len(expected):
        if any(action is None for action in actual[len(expected) :]):
            return "malformed", correct, "assistant submitted non-empty text"
        return "incorrect", correct, "trace contains extra assistant moves after puzzle completion"
    return "solved", correct, f"matched all {len(expected)} recorded player moves"


def progress_reward(correct: int, required: int, *, solved: bool) -> float:
    """Combine normalized prefix progress with an equally weighted solve bonus."""

    if required <= 0:
        raise ValueError("required moves must be positive")
    if not 0 <= correct <= required:
        raise ValueError("correct moves must be between zero and required moves")
    if solved and correct != required:
        raise ValueError("a solved puzzle must match every required move")
    progress = correct / required
    return PROGRESS_REWARD_WEIGHT * progress + (1 - PROGRESS_REWARD_WEIGHT) * float(solved)


class LichessPuzzleGrader(Grader):
    """Reward normalized prefix progress plus a terminal completion bonus."""

    async def _grade(
        self,
        grader_params: dict | None,
        trace: Trace,
        env: EnvironmentProtocol,
    ) -> GraderOutput:
        del env
        params: dict[str, Any] = grader_params or {}
        raw_expected = params.get("expected_player_moves", [])
        if (
            not raw_expected
            or not isinstance(raw_expected, list)
            or not all(isinstance(move, str) for move in raw_expected)
        ):
            return GraderOutput(score=0.0, reasoning="invalid expected_player_moves")

        expected = list(raw_expected)
        extraction = extract_actions(trace)
        actual = extraction.actions
        outcome, correct, reasoning = grade_actions(expected, actual)
        solved = outcome == "solved"
        malformed = outcome == "malformed"
        wrong_move = outcome == "incorrect"
        raw_max_tries = params.get("tries", DEFAULT_TRIES)
        max_tries = (
            raw_max_tries
            if isinstance(raw_max_tries, int)
            and not isinstance(raw_max_tries, bool)
            and raw_max_tries > 0
            else DEFAULT_TRIES
        )
        terminal_wrong_move_attempts = len(extraction.terminal_wrong_move_actions)
        wrong_move_attempts = (
            len(extraction.wrong_move_retry_actions) + terminal_wrong_move_attempts
        )
        remaining_tries = max(0, max_tries - wrong_move_attempts)
        progress = correct / len(expected)
        reward = progress_reward(correct, len(expected), solved=solved)
        artifacts = {
            "outcome": outcome,
            "solved": float(solved),
            "incorrect": float(outcome == "incorrect"),
            "incomplete": float(outcome == "incomplete"),
            "malformed": float(malformed),
            "wrong_move": float(wrong_move),
            "missing_action": float(outcome == "incomplete"),
            "correct_moves": correct,
            "required_moves": len(expected),
            "progress_fraction": progress,
            "completion_bonus": float(solved),
            "reward": reward,
            "reward_scheme": REWARD_SCHEME,
            "action_interface": extraction.interface,
            "format_retries": len(extraction.format_retry_reasons),
            "had_format_retry": float(bool(extraction.format_retry_reasons)),
            "format_retry_reasons": list(extraction.format_retry_reasons),
            "wrong_move_retries": len(extraction.wrong_move_retry_actions),
            "had_wrong_move_retry": float(bool(extraction.wrong_move_retry_actions)),
            "wrong_move_retry_actions": list(extraction.wrong_move_retry_actions),
            "terminal_wrong_move_attempts": terminal_wrong_move_attempts,
            "terminal_wrong_move_actions": list(extraction.terminal_wrong_move_actions),
            "wrong_move_attempts": wrong_move_attempts,
            "max_tries": max_tries,
            "remaining_tries": remaining_tries,
            "tries_exhausted": float(bool(extraction.terminal_wrong_move_actions)),
            "malformed_reason": extraction.malformed_reason if malformed else "",
            "text_submission": float(
                malformed and extraction.malformed_reason == "text_submission"
            ),
            "invalid_tool_name": float(malformed and extraction.malformed_reason == "wrong_tool"),
            "invalid_tool_arguments": float(
                malformed
                and extraction.malformed_reason
                in {"invalid_tool_json", "invalid_tool_arguments", "invalid_uci"}
            ),
            "multiple_tool_calls": float(
                malformed and extraction.malformed_reason == "multiple_tool_calls"
            ),
            "puzzle_rating": int(params.get("rating", 0)),
            "rating_bin": str(params.get("rating_bin", "unknown")),
            "length_bin": str(params.get("length_bin", "unknown")),
            "puzzle_id": str(params.get("puzzle_id", "")),
        }
        return GraderOutput(score=reward, reasoning=reasoning, artifacts=artifacts)
