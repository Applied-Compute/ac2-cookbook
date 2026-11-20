from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, DatasetSource, EvalConfig


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="geo3k-vlm")
    parser.add_argument("--model", default="gpt-5.6-luna")
    args = parser.parse_args()

    client = Client(project=args.project)
    run = await client.eval.run(
        EvalConfig(
            dataset=DatasetSource(dataset="geo3k-validation", num_tasks=8),
            max_parallel=8,
            name="geo3k-vlm",
            custom_harness=CustomHarnessConfig(
                model=ModelConfiguration(model=args.model),
            ),
        )
    )
    print(run.eval_id)


if __name__ == "__main__":
    asyncio.run(main())
