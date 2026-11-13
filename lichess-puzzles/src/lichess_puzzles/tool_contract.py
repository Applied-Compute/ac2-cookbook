"""Shared move-submission tool contract."""

import re
from typing import Any

ACTION_INTERFACE = "submit_move_squares_with_retries_v3"
SUBMIT_MOVE_TOOL = "submit_move"
QWEN36_TOOL_CALL_PARSER = "qwen3_coder"
QWEN36_REASONING_PARSER = "qwen3"
DEFAULT_TRIES = 3
SQUARE_PATTERN = re.compile(r"[a-h][1-8]")
PROMOTION_PIECES = frozenset({"q", "r", "b", "n"})


def parse_submit_move_arguments(arguments: Any) -> tuple[str | None, str]:
    """Convert the structured tool arguments to UCI or return a format-error code."""

    if not isinstance(arguments, dict):
        return None, "invalid_tool_arguments"
    keys = set(arguments)
    if not {"from_square", "to_square"} <= keys or not keys <= {
        "from_square",
        "to_square",
        "promotion",
    }:
        return None, "invalid_tool_arguments"

    from_square = arguments["from_square"]
    to_square = arguments["to_square"]
    if (
        not isinstance(from_square, str)
        or SQUARE_PATTERN.fullmatch(from_square) is None
        or not isinstance(to_square, str)
        or SQUARE_PATTERN.fullmatch(to_square) is None
    ):
        return None, "invalid_square"

    promotion = arguments.get("promotion")
    if promotion is not None and (
        not isinstance(promotion, str) or promotion not in PROMOTION_PIECES
    ):
        return None, "invalid_promotion"
    return f"{from_square}{to_square}{promotion or ''}", ""


def training_tool_call_parser(model: str) -> str:
    """Return the SGLang parser matching the selected Qwen model family."""

    return QWEN36_TOOL_CALL_PARSER if "Qwen3.6" in model else "qwen3"


def training_reasoning_parser(model: str) -> str | None:
    """Return the SGLang reasoning parser for supported thinking models."""

    return QWEN36_REASONING_PARSER if "Qwen3" in model else None
