from __future__ import annotations

import asyncio

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, DatasetSource, EvalConfig


async def main() -> None:
    client = Client(project="byoh-smoke-example")
    run = await client.eval.run(
        EvalConfig(
            dataset=DatasetSource(dataset="byoh-smoke-eval"),
            name="byoh-smoke-example",
            custom_harness=CustomHarnessConfig(
                model=ModelConfiguration(model="gpt-5.6-luna"),
            ),
        )
    )
    print(run.eval_id)


if __name__ == "__main__":
    asyncio.run(main())
