from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, DatasetSource, EvalConfig

from .relay import running_relay_id


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="byoh-agents-web-search-api")
    parser.add_argument("--dataset", default="byoh-web-search-eval")
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    client = Client(project=args.project)
    run = await client.eval.run(
        EvalConfig(
            grader="WebSearchGrader",
            dataset=DatasetSource(dataset=args.dataset, num_tasks=3),
            max_parallel=3,
            name="byoh-agents-web-search-api",
            custom_harness=CustomHarnessConfig(
                relay_deployment_id=running_relay_id(client),
                model=ModelConfiguration(model=args.model),
            ),
        )
    )
    print(run.eval_id)


if __name__ == "__main__":
    asyncio.run(main())
