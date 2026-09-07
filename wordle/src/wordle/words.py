"""Offline dictionary and Wordle's two-pass, duplicate-aware scoring."""

import re
from collections import Counter
from functools import cache
from importlib.resources import files

GREEN = "🟩"
YELLOW = "🟨"
WHITE = "⬜"
WORD_LENGTH = 5


def normalize_word(value: object) -> str | None:
    """Trim surrounding whitespace and accept only five ASCII letters."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    # Validate before uppercasing: Unicode case folding can produce ASCII letters.
    if re.fullmatch(r"[A-Za-z]{5}", value) is None:
        return None
    return value.upper()


@cache
def dictionary() -> frozenset[str]:
    return frozenset(files("wordle").joinpath("data/words.txt").read_text().splitlines())


def score_guess(base_word: str, guess: str) -> str:
    """Score normalized words, reserving greens before left-to-right yellows.

    Each occurrence in the answer can match at most one guessed letter.
    Dictionary membership is enforced by the environment, not this pure function.
    """
    if any(re.fullmatch(r"[A-Z]{5}", word) is None for word in (base_word, guess)):
        raise ValueError("Scoring requires two five-letter uppercase ASCII words.")
    tiles = [WHITE] * WORD_LENGTH
    remaining: Counter[str] = Counter()
    for i, (answer_letter, guess_letter) in enumerate(zip(base_word, guess, strict=True)):
        if answer_letter == guess_letter:
            tiles[i] = GREEN
        else:
            remaining[answer_letter] += 1
    for i, letter in enumerate(guess):
        if tiles[i] != GREEN and remaining[letter] > 0:
            tiles[i] = YELLOW
            remaining[letter] -= 1
    return "".join(tiles)
