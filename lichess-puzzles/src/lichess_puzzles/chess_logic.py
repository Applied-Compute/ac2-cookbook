"""Pure chess and output-parsing helpers shared by the runtime and dataset builder."""

from __future__ import annotations

import json
from dataclasses import dataclass

import chess
from ac2.runtime import Item, Message

OBSERVATION_FORMAT = "square_map_json_v3"


@dataclass(frozen=True)
class ValidatedPuzzle:
    """A Lichess line after legality checks and setup-move application."""

    source_fen: str
    puzzle_fen: str
    moves: tuple[str, ...]
    player_moves: tuple[str, ...]
    normalized_puzzle_fen: str


def validate_puzzle(source_fen: str, moves: list[str] | tuple[str, ...]) -> ValidatedPuzzle:
    """Validate a complete Lichess line and return the actual puzzle-start position.

    Lichess stores the position before the opponent's setup move. Therefore ``moves[0]``
    is applied before the position is shown, and player actions are at odd indices.
    """

    move_tuple = tuple(moves)
    if len(move_tuple) < 2 or len(move_tuple) % 2:
        raise ValueError(
            "a Lichess line must contain setup + player moves and end on a player move"
        )

    board = chess.Board(source_fen)
    puzzle_fen = ""
    for index, uci in enumerate(move_tuple):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"invalid UCI move at index {index}: {uci!r}") from exc
        if move not in board.legal_moves:
            raise ValueError(f"illegal move at index {index}: {uci!r} in {board.fen()!r}")
        board.push(move)
        if index == 0:
            puzzle_fen = board.fen()

    if not puzzle_fen:
        raise AssertionError("validated puzzle has no puzzle-start FEN")
    normalized = " ".join(puzzle_fen.split()[:4])
    return ValidatedPuzzle(
        source_fen=source_fen,
        puzzle_fen=puzzle_fen,
        moves=move_tuple,
        player_moves=move_tuple[1::2],
        normalized_puzzle_fen=normalized,
    )


def assistant_text(items: list[Item]) -> str | None:
    """Return the sole assistant message text, or ``None`` for an incomplete completion."""

    messages = [item for item in items if isinstance(item, Message) and item.role == "assistant"]
    if not messages:
        return None
    if len(messages) != 1:
        return ""
    content = messages[0].content
    if isinstance(content, str):
        return content
    blocks: list[str] = []
    for block in content:
        text = block.get("text")
        if isinstance(text, str):
            blocks.append(text)
    return "".join(blocks)


def render_square_map(board: chess.Board) -> str:
    """Render every square as deterministic JSON with explicit color and piece names."""

    squares: dict[str, str] = {}
    for file_index, file_name in enumerate(chess.FILE_NAMES):
        for rank in range(1, 9):
            piece = board.piece_at(chess.square(file_index, rank - 1))
            if piece is None:
                value = "empty"
            else:
                color = "white" if piece.color == chess.WHITE else "black"
                value = f"{color} {chess.piece_name(piece.piece_type)}"
            squares[f"{file_name}{rank}"] = value

    payload = {
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "squares": squares,
    }
    return json.dumps(payload, indent=2)


def board_observation(board: chess.Board) -> str:
    """Render the complete model-facing position in a fenced JSON block."""

    return f"```json\n{render_square_map(board)}\n```"


def position_prompt(
    fen: str,
    *,
    reasoning_budget: int | None = None,
    remaining_tries: int | None = None,
) -> str:
    """Render the user-visible observation without leaking rating, themes, or solution length."""

    board = chess.Board(fen)
    budget = (
        f"Suggested private reasoning budget: at most {reasoning_budget} tokens.\n"
        if reasoning_budget is not None
        else ""
    )
    tries = (
        f"Correctly formatted wrong-move tries remaining: {remaining_tries}.\n"
        if remaining_tries is not None
        else ""
    )
    return (
        f"{budget}{tries}"
        "Call submit_move exactly once with from_square, to_square, and promotion only when "
        "needed. Do not submit plain text.\n\n"
        f"Board:\n{board_observation(board)}"
    )
