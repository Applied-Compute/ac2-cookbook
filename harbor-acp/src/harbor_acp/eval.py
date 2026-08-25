from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, EvalConfig

from .relay import running_relay_id


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="harbor-acp")
    parser.add_argument("--dataset", default="harbor-hello-world")
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    client = Client(project=args.project)
    await client.eval.run(
        EvalConfig(
            orchestrator="HarborACPOrchestrator",
            grader="HarborRewardGrader",
            dataset=args.dataset,
            max_parallel=1,
            name="harbor-acp-hello-world",
            custom_harness=CustomHarnessConfig(
                relay_deployment_id=running_relay_id(client),
                model=ModelConfiguration(
                    model=args.model,
                    kwargs={"max_output_tokens": 8_192},
                ),
            ),
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
