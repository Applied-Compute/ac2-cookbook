from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, EvalConfig


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="tau2bench")
    parser.add_argument("--dataset", default="tau2bench-airline-test")
    parser.add_argument("--model", default="gpt-5.6-luna")
    args = parser.parse_args()

    client = Client(project=args.project)
    run = await client.eval.run(
        EvalConfig(
            dataset=args.dataset,
            name="tau2bench",
            custom_harness=CustomHarnessConfig(
                model=ModelConfiguration(model=args.model),
            ),
        )
    )
    print(run.eval_id)


if __name__ == "__main__":
    asyncio.run(main())
