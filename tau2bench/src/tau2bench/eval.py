from __future__ import annotations

import argparse
import asyncio

from ac2.sdk import Client, EvalConfig

from .agent import DOMAIN_ENVIRONMENT, Tau2Agent
from .dataloader import DOMAINS, Domain
from .graders import Tau2BenchGrader
from .user import Tau2BenchDefaultUser


def eval_config(domain: Domain = "airline") -> EvalConfig:
    return EvalConfig(
        agent=Tau2Agent(domain=domain),
        env=DOMAIN_ENVIRONMENT[domain](),
        grader=Tau2BenchGrader(),
        user=Tau2BenchDefaultUser(),
        dataset=f"tau2bench-{domain}-test",
        num_samples=1,
        max_parallel=8,
    )


CONFIG = eval_config()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="airline")
    args = parser.parse_args()
    await Client(project="tau2bench").eval.run(eval_config(args.domain))


if __name__ == "__main__":
    asyncio.run(main())
