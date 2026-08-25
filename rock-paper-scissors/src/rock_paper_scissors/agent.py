from __future__ import annotations

import os

from ac2.runtime import Agent, EnvironmentProtocol, Item, ModelConfiguration

MAX_ROUNDS = 40


def _debug() -> bool:
    return os.environ.get("RPS_DEBUG") == "1"


class RPSAgent(Agent):
    description = "Plays rock-paper-scissors against a mystery opponent."
    model_configuration = ModelConfiguration(model="gpt-5-mini")
    system_prompt = (
        f"You are playing rock-paper-scissors against a mystery opponent for {MAX_ROUNDS} rounds.\n"
        "\n"
        "Tools:\n"
        "- play_move(move): play one round. `move` must be 'rock', 'paper', or 'scissors'. "
        "  Returns the opponent's move, the round result, the running score, and rounds left.\n"
        "- run_python(code): execute Python to analyze the history and plan strategy.\n"
        "\n"
        f"Play all {MAX_ROUNDS} rounds. Use the round-by-round feedback from play_move "
        "(and run_python if helpful) to detect patterns in the opponent's behavior and "
        f"maximize your win rate. After {MAX_ROUNDS} rounds the game is over; stop calling "
        "play_move."
    )

    async def on_completion(self, items: list[Item], env: EnvironmentProtocol) -> None:
        if not _debug():
            return
        for item in items:
            cls = type(item).__name__
            if cls == "Message":
                role = getattr(item, "role", "?")
                content = getattr(item, "content", "")
                print(f"\n\033[36m[{role}]\033[0m {content}", flush=True)
            elif cls == "FunctionCall":
                name = getattr(item, "name", "?")
                args = getattr(item, "arguments", "")
                print(f"\n\033[33m[tool call] {name}({args})\033[0m", flush=True)
            else:
                print(f"\n\033[35m[{cls}]\033[0m {item!r}", flush=True)
