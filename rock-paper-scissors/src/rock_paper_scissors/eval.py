from __future__ import annotations

import argparse
import asyncio

from ac2.sdk import Client, DatasetSource, EvalConfig

DATASET = "rock-paper-scissors"


CONFIG = EvalConfig(
    agent="RPSAgent",
    env="RPSEnvironment",
    grader="WinrateGrader",
    dataset=DatasetSource(dataset=DATASET, num_tasks=1),
    num_samples=1,
    max_parallel=1,
)


async def main(*, local: bool = False) -> None:
    await Client(project="rock-paper-scissors").eval.run(CONFIG, local=local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(local=args.local))
