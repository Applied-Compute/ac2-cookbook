from __future__ import annotations

import argparse
import asyncio

from ac2.sdk import Client, DatasetSource, EvalConfig

from .agent import DapoMathAgent
from .environment import DapoMathCheckEnvironment
from .grader import DapoMathCheckGrader
from .user import DapoMathCheckUser

DATASET = "dapo-math-check-eval256"


def eval_config(model: str = "gpt-5-mini") -> EvalConfig:
    return EvalConfig(
        agent=DapoMathAgent(model=model),
        env=DapoMathCheckEnvironment(),
        grader=DapoMathCheckGrader(),
        user=DapoMathCheckUser(),
        dataset=DatasetSource(dataset=DATASET, num_tasks=10),
        num_samples=1,
        max_parallel=4,
    )


CONFIG = eval_config()


async def main(*, model: str = "gpt-5-mini") -> None:
    await Client(project="dapo-math-check").eval.run(eval_config(model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()
    asyncio.run(main(model=args.model))
