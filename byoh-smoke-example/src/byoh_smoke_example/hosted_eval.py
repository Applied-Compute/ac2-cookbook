from __future__ import annotations

from ac2.runtime import ModelConfiguration
from ac2.sdk import Client, CustomHarnessConfig, EvalConfig

from byoh_smoke_example.harness import SmokeHarness


def main() -> None:
    client = Client(project="byoh-smoke-example")
    run = client.eval.run(
        EvalConfig(
            harness=SmokeHarness(),
            dataset="byoh-smoke-eval",
            name="byoh-smoke-example-hosted",
            custom_harness=CustomHarnessConfig(
                model=ModelConfiguration(model="gpt-5.6-luna"),
            ),
        )
    )
    print(run.eval_id)


if __name__ == "__main__":
    main()
