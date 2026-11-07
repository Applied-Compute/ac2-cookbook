from __future__ import annotations

import asyncio

from ac2.sdk import Client, DatasetSource, EvalConfig

from .agent import RPSAgent
from .environment import RPSEnvironment
from .grader import WinrateGrader

DATASET = "rock-paper-scissors"


CONFIG = EvalConfig(
    agent=RPSAgent(),
    env=RPSEnvironment(),
    grader=WinrateGrader(),
    dataset=DatasetSource(dataset=DATASET, num_tasks=1),
    num_samples=1,
    max_parallel=1,
)


async def main() -> None:
    await Client(project="rock-paper-scissors").eval.run(CONFIG)


if __name__ == "__main__":
    asyncio.run(main())
