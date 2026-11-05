"""Deterministic replay environment for a single Lichess puzzle."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

import chess
from ac2.runtime import Environment, FunctionCall, FunctionCallOutput, Item, tool
from ac2.tracing import annotate
from pydantic import Field

from .chess_logic import (
    assistant_text,
    board_observation,
    position_prompt,
    validate_puzzle,
)
from .tool_contract import DEFAULT_TRIES, SUBMIT_MOVE_TOOL, parse_submit_move_arguments

Outcome = Literal["not_started", "active", "solved", "incorrect", "incomplete", "malformed"]


class LichessPuzzleEnvironment(Environment):
    """Replay the recorded line with free format retries and bounded wrong-move retries."""

    def __init__(self) -> None:
        self.board: chess.Board | None = None
        self.moves: tuple[str, ...] = ()
        self.next_move_index = 1
        self.correct_moves = 0
        self.max_tries = DEFAULT_TRIES
        self.remaining_tries = DEFAULT_TRIES
        self.outcome: Outcome = "not_started"
        self.failure_reason = ""
        self.failure_type = ""
        self.last_response: str | None = None
        self.puzzle_id = ""
        self.reasoning_budget: int | None = None
        self.metadata: dict[str, Any] = {}

    async def setup(self, env_params: dict) -> None:
        source_fen = str(env_params["source_fen"])
        raw_moves = env_params["moves"]
        if not isinstance(raw_moves, list) or not all(isinstance(move, str) for move in raw_moves):
            raise ValueError("env_params['moves'] must be a list of UCI strings")
        puzzle = validate_puzzle(source_fen, raw_moves)

        self.board = chess.Board(puzzle.puzzle_fen)
        self.moves = puzzle.moves
        self.next_move_index = 1
        self.correct_moves = 0
        raw_tries = env_params.get("tries", DEFAULT_TRIES)
        if isinstance(raw_tries, bool) or not isinstance(raw_tries, int) or raw_tries <= 0:
            raise ValueError("env_params['tries'] must be a positive integer")
        self.max_tries = raw_tries
        self.remaining_tries = raw_tries
        self.outcome = "active"
        self.failure_reason = ""
        self.failure_type = ""
        self.last_response = None
        self.puzzle_id = str(env_params.get("puzzle_id", ""))
        budget = env_params.get("reasoning_budget")
        self.reasoning_budget = int(budget) if budget is not None else None
        raw_metadata = env_params.get("metadata", {})
        self.metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

        annotate(
            {
                "puzzle_id": self.puzzle_id,
                "puzzle_rating": self.metadata.get("rating"),
                "rating_bin": self.metadata.get("rating_bin"),
                "solution_player_moves": len(puzzle.player_moves),
                "length_bin": self.metadata.get("length_bin"),
                "dataset_split": self.metadata.get("split"),
                "max_tries": self.max_tries,
            }
        )

    async def step(self, items: list[Item]) -> tuple[list[FunctionCallOutput], bool]:
        if self.outcome != "active":
            return [], True
        if self.board is None:
            raise RuntimeError("environment has not been set up")

        calls = [item for item in items if isinstance(item, FunctionCall)]
        self.last_response = assistant_text(items)
        if not calls:
            if self.last_response is None:
                self._fail(
                    "incomplete",
                    "completion contained no submit_move tool call",
                    "missing_tool_call",
                )
            else:
                self._fail(
                    "malformed",
                    "assistant submitted text instead of calling submit_move",
                    "text_submission",
                )
            return [], True
        if len(calls) != 1:
            return self._format_error_outputs(
                calls,
                "multiple_tool_calls",
                f"Expected exactly one submit_move call, but received {len(calls)}.",
            ), False

        call = calls[0]
        if call.name != SUBMIT_MOVE_TOOL:
            return self._format_error_outputs(
                calls,
                "wrong_tool",
                f"Wrong tool name: expected {SUBMIT_MOVE_TOOL!r}, but received {call.name!r}.",
            ), False
        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            return self._format_error_outputs(
                calls,
                "invalid_tool_json",
                "The submit_move arguments were not valid JSON.",
            ), False
        action, format_error = parse_submit_move_arguments(arguments)
        if action is None:
            return self._format_error_outputs(
                calls,
                format_error,
                self._format_error_message(format_error),
            ), False

        output = await self._apply_move(action)
        return [FunctionCallOutput(call_id=call.call_id, output=output)], self.outcome != "active"

    @tool(
        "Submit one chess move using its origin square, destination square, and optional lowercase "
        "promotion piece. Format errors do not consume tries. A wrong move consumes one try and "
        "leaves the board unchanged while tries remain."
    )
    async def submit_move(
        self,
        from_square: Annotated[
            str,
            Field(
                description="The lowercase origin square, such as 'e2'.",
                pattern=r"^[a-h][1-8]$",
            ),
        ],
        to_square: Annotated[
            str,
            Field(
                description="The lowercase destination square, such as 'e4'.",
                pattern=r"^[a-h][1-8]$",
            ),
        ],
        promotion: Annotated[
            Literal["q", "r", "b", "n"] | None,
            Field(
                description=(
                    "For a pawn promotion only, the lowercase promoted piece: q, r, b, or n. "
                    "Omit this field for all non-promotion moves."
                )
            ),
        ] = None,
    ) -> str:
        """Validate one structured submission and encode its result."""

        action, format_error = parse_submit_move_arguments(
            {
                "from_square": from_square,
                "to_square": to_square,
                "promotion": promotion,
            }
        )
        if action is None:
            return self._format_error_output(
                format_error,
                self._format_error_message(format_error),
            )
        return await self._apply_move(action)

    async def _apply_move(self, move: str) -> str:
        """Apply one well-formatted UCI move and encode the next observation."""

        if self.outcome != "active" or self.board is None:
            raise RuntimeError("no active puzzle")
        expected = self.moves[self.next_move_index]
        if move != expected:
            self.remaining_tries -= 1
            tries_exhausted = self.remaining_tries == 0
            if tries_exhausted:
                self._fail("incorrect", f"expected {expected}, received {move}", "wrong_move")
            return self._tool_output(
                status="incorrect" if tries_exhausted else "wrong_move",
                move=move,
                max_tries=self.max_tries,
                remaining_tries=self.remaining_tries,
                position=board_observation(self.board),
                message=(
                    "The submitted move did not match the recorded continuation. "
                    + (
                        "No tries remain; the puzzle is over."
                        if tries_exhausted
                        else "The board is unchanged; try again."
                    )
                ),
            )

        board_move = chess.Move.from_uci(expected)
        if board_move not in self.board.legal_moves:
            raise RuntimeError(f"recorded player move {expected!r} became illegal")
        self.board.push(board_move)
        self.correct_moves += 1
        self.next_move_index += 1

        if self.next_move_index == len(self.moves):
            self.outcome = "solved"
            annotate(
                {
                    "puzzle_outcome": self.outcome,
                    "puzzle_solved": True,
                    "correct_moves": self.correct_moves,
                    "remaining_tries": self.remaining_tries,
                }
            )
            return self._tool_output(
                status="solved",
                move=move,
                max_tries=self.max_tries,
                remaining_tries=self.remaining_tries,
                position=board_observation(self.board),
                message="Puzzle complete.",
            )

        opponent_reply = chess.Move.from_uci(self.moves[self.next_move_index])
        if opponent_reply not in self.board.legal_moves:
            raise RuntimeError(f"recorded opponent reply {opponent_reply.uci()!r} became illegal")
        self.board.push(opponent_reply)
        self.next_move_index += 1
        return self._tool_output(
            status="correct",
            move=move,
            opponent_reply=opponent_reply.uci(),
            max_tries=self.max_tries,
            remaining_tries=self.remaining_tries,
            position=board_observation(self.board),
            suggested_reasoning_budget=self.reasoning_budget,
            message="Correct. The recorded opponent reply has been played; submit the next move.",
        )

    def initial_position_prompt(self) -> str:
        """Render the initial observation from live state, not stored task text."""

        if self.outcome != "active" or self.board is None:
            raise RuntimeError("no initial position is available")
        return position_prompt(
            self.board.fen(),
            reasoning_budget=self.reasoning_budget,
            remaining_tries=self.remaining_tries,
        )

    def mark_incomplete(self, reason: str) -> None:
        if self.outcome == "active":
            self._fail("incomplete", reason, "orchestrator_limit")

    def _format_error_outputs(
        self,
        calls: list[FunctionCall],
        failure_type: str,
        reason: str,
    ) -> list[FunctionCallOutput]:
        output = self._format_error_output(failure_type, reason)
        return [FunctionCallOutput(call_id=call.call_id, output=output) for call in calls]

    def _format_error_output(self, failure_type: str, reason: str) -> str:
        if self.board is None:
            raise RuntimeError("environment has not been set up")
        return self._tool_output(
            status="format_error",
            error_type=failure_type,
            max_tries=self.max_tries,
            remaining_tries=self.remaining_tries,
            message=(
                f"Formatting error: {reason} The puzzle is still active. "
                "Call submit_move again with corrected arguments."
            ),
            position=board_observation(self.board),
        )

    @staticmethod
    def _format_error_message(failure_type: str) -> str:
        if failure_type == "invalid_square":
            return "from_square and to_square must each match one lowercase square such as 'e2'."
        if failure_type == "invalid_promotion":
            return "promotion must be omitted or set to one of: q, r, b, n."
        return (
            "submit_move requires from_square and to_square, plus only the optional promotion "
            "field."
        )

    @staticmethod
    def _tool_output(**payload: Any) -> str:
        position = payload.pop("position", None)
        header = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        if position is None:
            return header
        return f"{header}\n\nBoard:\n{position}"

    def _fail(
        self,
        outcome: Literal["incorrect", "incomplete", "malformed"],
        reason: str,
        failure_type: str,
    ) -> None:
        self.outcome = outcome
        self.failure_reason = reason
        self.failure_type = failure_type
        annotate(
            {
                "puzzle_outcome": outcome,
                "puzzle_solved": False,
                "correct_moves": self.correct_moves,
                "max_tries": self.max_tries,
                "remaining_tries": self.remaining_tries,
                "wrong_move_attempts": self.max_tries - self.remaining_tries,
                "failure_reason": reason,
                "failure_type": failure_type,
            }
        )
