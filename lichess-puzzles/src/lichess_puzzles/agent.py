"""Trainable chess-puzzle policy."""

from typing import ClassVar

from ac2.runtime import Agent, ModelConfiguration

from .tool_contract import SUBMIT_MOVE_TOOL

DEFAULT_EVAL_MODEL = "gpt-5.6-luna"
DEFAULT_MAX_OUTPUT_TOKENS = 4_096
QWEN36_MODEL = "Qwen/Qwen3.6-35B-A3B"
QWEN36_MAX_OUTPUT_TOKENS = 8_192

SYSTEM_PROMPT = """\
You solve chess puzzles one move at a time. Each position is a JSON object. `side_to_move` is
`"white"` or `"black"`. `squares` contains all 64 named squares from `a1` through `h8`; each value
is `"empty"` or an explicit piece such as `"white knight"` or `"black queen"`. Analyze the
position privately, then call `submit_move` exactly once with your move in structured square
notation: for example, use `from_square="e2", to_square="e4"`; for a promotion, also set
`promotion="q"`, `"r"`, `"b"`, or `"n"`. Omit promotion for every non-promotion move.
Do not submit a move as plain text. If the tool reports a formatting error, correct the arguments
and call it again; format errors do not consume a try. A correctly formatted move that does not
match the recorded continuation consumes one try but leaves the board unchanged while tries remain,
so use the returned board and try again. If your move is correct, the recorded opponent reply will
be played automatically and the tool result will contain the resulting board position in the same
format. Continue calling `submit_move` until the puzzle is complete.
"""


class LichessPuzzleAgent(Agent):
    """Policy used for both baseline evaluation and GRPO training."""

    description = "Chooses one UCI chess move at each step of a Lichess puzzle."
    system_prompt = SYSTEM_PROMPT
    allowed_tools: ClassVar[list[str]] = [SUBMIT_MOVE_TOOL]
    model_configuration = ModelConfiguration(
        model=DEFAULT_EVAL_MODEL,
        api_type="responses",
        kwargs={
            "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": "low"},
            "tool_choice": "required",
        },
    )

class LichessPuzzleQwen36Agent(LichessPuzzleAgent):
    """Self-hosted Qwen3.6 baseline with an expanded reasoning budget."""

    description = "Chooses one UCI chess move using the Qwen3.6-35B-A3B baseline."
    model_configuration = ModelConfiguration(
        model=QWEN36_MODEL,
        api_type="completions",
        kwargs={
            "max_tokens": QWEN36_MAX_OUTPUT_TOKENS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            # SGLang's qwen3_coder parser can repeat or corrupt Qwen3.6 calls when
            # tool_choice="required". Auto mode still exposes only submit_move,
            # while the system and position prompts require exactly one call.
            "tool_choice": "auto",
        },
    )
