from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig

_COMMON = dict(
    model="Qwen/Qwen3-4B",
    method="opsd",
    n_training_replicas=4,
    samples_per_problem=1,
    problem_batch_size=8,
    ac2_train_dataset="dapo-math-check-opsd-train",
)

CONFIGS = {
    "offline": TrainingConfig(**_COMMON, opsd_rollout_mode="offline", n_inference_replicas=0),
    "one_step": TrainingConfig(**_COMMON, opsd_rollout_mode="one_step", n_inference_replicas=4),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=CONFIGS, default="offline")
    args = parser.parse_args()
    Client(project="dapo-math-check").train.run(CONFIGS[args.mode])


if __name__ == "__main__":
    main()
