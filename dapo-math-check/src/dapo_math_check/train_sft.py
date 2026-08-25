from __future__ import annotations

from ac2.sdk import Client, TrainingConfig

CONFIG = TrainingConfig(
    model="Qwen/Qwen3-4B",
    method="sft",
    n_training_replicas=4,
    n_inference_replicas=0,
    samples_per_problem=1,
    problem_batch_size=8,
    ac2_train_dataset="dapo-math-check-opsd-train",
)


def main() -> None:
    Client(project="dapo-math-check").train.run(CONFIG)


if __name__ == "__main__":
    main()
