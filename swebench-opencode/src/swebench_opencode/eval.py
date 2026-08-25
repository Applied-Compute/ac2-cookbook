from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, EvalConfig

from .relay import running_relay_id


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="swebench-opencode")
    parser.add_argument("--dataset", default="swebench-opencode-smoke")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-parallel", type=int, default=1)
    args = parser.parse_args()

    client = Client(project=args.project)
    await client.eval.run(
        EvalConfig(
            orchestrator="SwebenchOpenCodeOrchestrator",
            grader="SwebenchVerifiedGrader",
            dataset=args.dataset,
            max_parallel=args.max_parallel,
            name="swebench-opencode",
            custom_harness=CustomHarnessConfig(
                relay_deployment_id=running_relay_id(client),
                model=ModelConfiguration(
                    model=args.model,
                    kwargs={"max_output_tokens": 32_768},
                ),
            ),
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
