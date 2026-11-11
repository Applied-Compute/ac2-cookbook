from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, DatasetSource, EvalConfig

from .grader import AnswerGrader


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="byoh-smoke-example")
    parser.add_argument("--dataset", default="byoh-smoke-eval")
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    client = Client(project=args.project)
    run = await client.eval.run(
        EvalConfig(
            grader=AnswerGrader(),
            dataset=DatasetSource(dataset=args.dataset, num_tasks=3),
            max_parallel=3,
            name="byoh-smoke-example",
            custom_harness=CustomHarnessConfig(
                model=ModelConfiguration(model=args.model),
            ),
        )
    )
    print(run.eval_id)


if __name__ == "__main__":
    asyncio.run(main())
