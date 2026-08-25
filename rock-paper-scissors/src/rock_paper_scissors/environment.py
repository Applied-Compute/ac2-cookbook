from __future__ import annotations

import ast
import io
import random
from contextlib import redirect_stdout
from typing import Annotated, Literal

from ac2.runtime import Environment, FunctionCallOutput, Item, tool
from pydantic import Field

from .agent import MAX_ROUNDS, _debug


class RPSEnvironment(Environment):
    def __init__(self) -> None:
        self.history: list[dict] = []
        self.wins: int = 0
        self.losses: int = 0
        self.ties: int = 0
        self.max_rounds: int = MAX_ROUNDS
        self._py_ns: dict = {"__builtins__": __builtins__}

    async def setup(self, env_params: dict | None = None) -> None:
        self.history = []
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.max_rounds = (env_params or {}).get("max_rounds", MAX_ROUNDS)
        self._py_ns = {"__builtins__": __builtins__}

    async def step(self, items: list[Item]) -> tuple[list[FunctionCallOutput], bool]:
        outputs, done = await super().step(items)
        if _debug():
            for out in outputs:
                content = getattr(out, "output", out)
                print(f"\n\033[32m[tool output]\033[0m {content}", flush=True)
        return outputs, done

    def _opponent_move(self) -> str:
        if not self.history:
            return random.choice(["rock", "paper", "scissors"])
        last_agent_move = self.history[-1]["agent"]
        beats = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
        return beats[last_agent_move]

    @staticmethod
    def _judge(agent_move: str, opponent_move: str) -> Literal["win", "loss", "tie"]:
        if agent_move == opponent_move:
            return "tie"
        beats = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
        return "win" if beats[agent_move] == opponent_move else "loss"

    @tool("Play one round of rock-paper-scissors against the mystery opponent.")
    async def play_move(
        self,
        move: Annotated[
            Literal["rock", "paper", "scissors"],
            Field(description="Your move for this round."),
        ],
    ) -> str:
        if len(self.history) >= self.max_rounds:
            return f"Game over: all {self.max_rounds} rounds have been played. Stop calling play_move."

        opponent = self._opponent_move()
        result = self._judge(move, opponent)
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:
            self.ties += 1

        round_num = len(self.history) + 1
        self.history.append({"round": round_num, "agent": move, "opponent": opponent, "result": result})
        rounds_left = self.max_rounds - round_num
        return (
            f"Round {round_num}: you={move}, opponent={opponent}, result={result}. "
            f"Score: {self.wins}W/{self.losses}L/{self.ties}T. Rounds left: {rounds_left}."
        )

    @tool(
        "Execute Python code and return the result. Prints go to stdout; the last "
        "expression value is also returned."
    )
    async def run_python(
        self,
        code: Annotated[str, Field(description="Python code to execute.")],
    ) -> str:
        buf = io.StringIO()
        try:
            tree = ast.parse(code, "<agent>", "exec")
            trailing_expr: ast.Expression | None = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = tree.body.pop()
                trailing_expr = ast.Expression(body=last.value)
                ast.copy_location(trailing_expr, last)
            with redirect_stdout(buf):
                exec(compile(tree, "<agent>", "exec"), self._py_ns)  # noqa: S307
                if trailing_expr is not None:
                    last_val = eval(compile(trailing_expr, "<agent>", "eval"), self._py_ns)  # noqa: S307
                    if last_val is not None:
                        buf.write(repr(last_val) + "\n")
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"
        output = buf.getvalue()
        if len(output) > 10_000:
            return output[:10_000] + "\n... (truncated)"
        return output or "(no output)"
