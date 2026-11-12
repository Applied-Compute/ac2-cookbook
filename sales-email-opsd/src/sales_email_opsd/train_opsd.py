from __future__ import annotations

import argparse
from typing import Literal

from ac2.sdk import Client, DatasetSource, TrainingConfig

from .agent import BASE_MODEL
from .dataset import TRAIN_DATASET

Mode = Literal["offline", "one_step"]


def training_config(
    mode: Mode = "one_step",
    *,
    model: str = BASE_MODEL,
    steps: int = 20,
    smoke: bool = False,
    dataset: str = TRAIN_DATASET,
) -> TrainingConfig:
    if mode not in {"offline", "one_step"}:
        raise ValueError(f"Unsupported OPSD mode: {mode}")
    if steps < 1:
        raise ValueError("steps must be positive")
    return TrainingConfig(
        model=model,
        method="opsd",
        opsd_rollout_mode=mode,
        n_training_replicas=4,
        n_inference_replicas=0 if mode == "offline" else 4,
        samples_per_problem=1,
        problem_batch_size=8,
        num_train_steps=1 if smoke else steps,
        seed=42,
        priority="low",
        ac2_train_dataset=DatasetSource(dataset=dataset, num_tasks=10 if smoke else None),
    )


CONFIGS = {mode: training_config(mode) for mode in ("offline", "one_step")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit an OPSD run; all modes allocate training compute.")
    parser.add_argument("--project", default="sales-email-opsd")
    parser.add_argument("--mode", choices=CONFIGS, default="one_step")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="One training step over at most ten tasks")
    parser.add_argument("--dataset", default=TRAIN_DATASET, help="Hinted replay dataset to train on")
    args = parser.parse_args()
    config = training_config(
        args.mode, model=args.model, steps=args.steps, smoke=args.smoke, dataset=args.dataset
    )
    print(Client(project=args.project).train.run(config))


if __name__ == "__main__":
    main()
