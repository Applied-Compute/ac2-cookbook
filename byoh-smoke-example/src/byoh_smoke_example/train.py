from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig, TrainingCustomHarnessConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    client = Client(project="byoh-smoke-example")
    run = client.train.run(
        TrainingConfig(
            model="Qwen/Qwen3-4B",
            n_inference_replicas=1,
            num_train_steps=args.steps,
            samples_per_problem=2,
            problem_batch_size=1,
            global_sampling_concurrency=2,
            ac2_train_dataset="byoh-smoke-eval",
            custom_harness=TrainingCustomHarnessConfig(),
            eval_mode="off",
            name="byoh-smoke-example",
        )
    )
    print(run.train_id)


if __name__ == "__main__":
    main()
