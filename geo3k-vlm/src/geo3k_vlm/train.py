from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig, TrainingCustomHarnessConfig


def config(train_dataset: str, eval_dataset: str, steps: int) -> TrainingConfig:
    return TrainingConfig(
        model="Qwen/Qwen3.5-9B-VL",
        n_training_replicas=1,
        n_inference_replicas=1,
        num_train_steps=steps,
        samples_per_problem=4,
        problem_batch_size=8,
        ac2_train_dataset=train_dataset,
        ac2_eval_dataset=eval_dataset,
        custom_harness=TrainingCustomHarnessConfig(),
        max_response_len=8192,
        eval_interval=steps,
        eval_samples_per_problem=1,
        name="geo3k-vlm",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="geo3k-vlm")
    parser.add_argument("--train-dataset", default="geo3k-train")
    parser.add_argument("--eval-dataset", default="geo3k-validation")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    client = Client(project=args.project)
    run = client.train.run(config(args.train_dataset, args.eval_dataset, args.steps))
    print(run.train_id)


if __name__ == "__main__":
    main()
