from __future__ import annotations

import asyncio

from ac2.sdk import Client, EvalConfig


CONFIG = EvalConfig(
    agent="Tau2AirlineAgent",
    env="AirlineEnvironment",
    grader="Tau2BenchGrader",
    user="Tau2BenchDefaultUser",
    dataset="tau2bench-airline-test",
    num_samples=1,
    max_parallel=8,
)


async def main() -> None:
    await Client(project="tau2bench").eval.run(CONFIG)


if __name__ == "__main__":
    asyncio.run(main())
