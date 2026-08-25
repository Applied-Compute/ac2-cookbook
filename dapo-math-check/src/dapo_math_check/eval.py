from __future__ import annotations

import argparse
import asyncio

from ac2.sdk import Client, DatasetSource, EvalConfig

DATASET = "dapo-math-check-eval256"


CONFIG = EvalConfig(
    agent="DapoMathAgent",
    env="DapoMathCheckEnvironment",
    grader="DapoMathCheckGrader",
    user="DapoMathCheckUser",
    dataset=DatasetSource(dataset=DATASET, num_tasks=10),
    num_samples=1,
    max_parallel=4,
)


async def main(*, local: bool = False) -> None:
    await Client(project="dapo-math-check").eval.run(CONFIG, local=local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(local=args.local))
