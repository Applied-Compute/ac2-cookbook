from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, DatasetSource, EvalConfig


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="swebench-opencode")
    parser.add_argument("--dataset", default="swebench-opencode-smoke")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--backend", choices=["dispatch", "modal"], default="dispatch")
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--name", default="swebench-opencode")
    args = parser.parse_args()

    client = Client(project=args.project)
    await client.eval.run(
        EvalConfig(
            orchestrator="SwebenchOpenCodeOrchestrator",
            grader="SwebenchVerifiedGrader",
            dataset=DatasetSource(dataset=args.dataset, num_tasks=args.num_tasks),
            max_parallel=args.max_parallel,
            backend=args.backend,
            name=args.name,
            custom_harness=CustomHarnessConfig(
                model=ModelConfiguration(
                    model=args.model,
                    kwargs={"max_output_tokens": 32_768},
                ),
            ),
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
