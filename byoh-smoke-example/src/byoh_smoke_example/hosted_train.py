from __future__ import annotations

import argparse

from ac2.sdk import Client, ExperimentConfig, RunConfig
from ac2.sdk.run_config import EvalConfig, StepConfig, TasksConfig, TrainDatasetConfig

from byoh_smoke_example.harness import SmokeHarness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    client = Client(project="byoh-smoke-example")
    run = client.train.run(
        RunConfig(
            model="Qwen/Qwen3-4B",
            name="byoh-smoke-example-hosted",
            experiment_cfg=ExperimentConfig(
                step=StepConfig(
                    num_train_steps=args.steps,
                    samples_per_problem=2,
                    problem_batch_size=1,
                ),
                eval=EvalConfig(mode="off"),
                tasks=TasksConfig(
                    datasets=(TrainDatasetConfig(ac2_dataset_name="byoh-smoke-eval"),),
                    harness=SmokeHarness(),
                ),
            ),
        )
    )
    print(run.train_id)


if __name__ == "__main__":
    main()
