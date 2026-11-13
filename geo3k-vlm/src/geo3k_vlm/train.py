from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig, TrainingCustomHarnessConfig


def config(steps: int) -> TrainingConfig:
    return TrainingConfig(
        model="Qwen/Qwen3.5-9B-VL",
        n_training_replicas=1,
        n_inference_replicas=1,
        num_train_steps=steps,
        samples_per_problem=4,
        problem_batch_size=8,
        ac2_train_dataset="geo3k-train",
        custom_harness=TrainingCustomHarnessConfig(),
        max_response_len=8192,
        eval_mode="off",
        name="geo3k-vlm",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="geo3k-vlm")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    client = Client(project=args.project)
    run = client.train.run(config(args.steps))
    print(run.train_id)


if __name__ == "__main__":
    main()
