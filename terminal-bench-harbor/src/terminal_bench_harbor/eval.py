from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, DatasetSource, EvalConfig


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="terminal-bench-harbor")
    parser.add_argument("--dataset", default="terminal-bench")
    parser.add_argument("--model", default="gpt-5.6-luna")
    args = parser.parse_args()

    client = Client(project=args.project)
    run = await client.eval.run(
        EvalConfig(
            dataset=DatasetSource(dataset=args.dataset, num_tasks=4),
            max_parallel=4,
            max_retries=0,
            name="terminal-bench-harbor",
            custom_harness=CustomHarnessConfig(
                model=ModelConfiguration(
                    model=args.model,
                    kwargs={"max_output_tokens": 8_192},
                ),
            ),
        )
    )
    print(run.eval_id)


if __name__ == "__main__":
    asyncio.run(main())
