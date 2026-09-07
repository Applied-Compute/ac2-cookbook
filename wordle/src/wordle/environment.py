from typing import Annotated

from ac2.runtime import Environment, FunctionCall, FunctionCallOutput, Item, tool
from pydantic import Field

from .words import GREEN, WORD_LENGTH, dictionary, normalize_word, score_guess

MAX_GUESSES = 6
INVALID_GUESS = "invalid guess"


class WordleEnvironment(Environment):
    def __init__(self) -> None:
        self._base_word = ""
        self.guess_count = 0
        self.solved = False
        self.terminated_reason = ""

    async def setup(self, env_params: dict) -> None:
        word = normalize_word(env_params.get("base_word"))
        if word is None or word not in dictionary():
            raise ValueError("base_word must be a five-letter word in the dictionary.")
        self._base_word = word
        self.guess_count = 0
        self.solved = False
        self.terminated_reason = ""

    @property
    def is_terminated(self) -> bool:
        return bool(self.terminated_reason)

    def _record_attempt(self) -> None:
        self.guess_count += 1
        if self.guess_count >= MAX_GUESSES:
            self.terminated_reason = "exhausted"

    async def step(self, items: list[Item]) -> tuple[list[FunctionCallOutput], bool]:
        if self.is_terminated:
            return [], True
        calls = [item for item in items if isinstance(item, FunctionCall)]
        if not calls:
            self.terminated_reason = "abandoned"
            return [], True
        outputs = []
        # Sequential dispatch keeps multiple calls from overrunning the budget.
        for call in calls:
            before = self.guess_count
            output = await self.call_tool(call)
            if self.guess_count == before:
                # Malformed JSON/arguments and hallucinated tools also consume
                # an attempt, so invalid tool calls cannot loop indefinitely.
                self._record_attempt()
                output = FunctionCallOutput(call_id=call.call_id, output=INVALID_GUESS)
            outputs.append(output)
            if self.is_terminated:
                break
        return outputs, self.is_terminated

    @tool("Guess a five-letter dictionary word. Returns 🟩/🟨/⬜ or 'invalid guess'. Six attempts maximum.")
    async def check_answer(
        self, guess: Annotated[str, Field(description="Your five-letter word guess.")],
    ) -> str:
        if self.is_terminated:
            return INVALID_GUESS
        self._record_attempt()
        word = normalize_word(guess)
        if word is None or word not in dictionary():
            return INVALID_GUESS
        feedback = score_guess(self._base_word, word)
        if feedback == GREEN * WORD_LENGTH:
            self.solved = True
            self.terminated_reason = "correct"
        return feedback
