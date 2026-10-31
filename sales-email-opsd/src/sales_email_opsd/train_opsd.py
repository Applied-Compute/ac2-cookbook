from __future__ import annotations

import argparse
from typing import Literal

from ac2.sdk import Client, DatasetSource, TrainingConfig

from .agent import BASE_MODEL, SalesEmailAgent
from .dataset import EVAL_DATASET, ONLINE_TRAIN_DATASET, TRAIN_DATASET
from .environment import SalesEmailEnvironment
from .grader import EmDashGrader
from .hinter import JUDGE_MODEL, SalesEmailJudgeHinter
from .user import ReplaySalesEmailUser, SalesEmailUser

Mode = Literal["offline", "one_step", "online"]
HINTER_PATH = f"{SalesEmailJudgeHinter.__module__}.{SalesEmailJudgeHinter.__name__}"


def training_config(
    mode: Mode = "online",
    *,
    model: str = BASE_MODEL,
    steps: int = 20,
    smoke: bool = False,
    replay_user: bool = False,
    token_judge: str | None = JUDGE_MODEL,
) -> TrainingConfig:
    if mode not in {"offline", "one_step", "online"}:
        raise ValueError(f"Unsupported OPSD mode: {mode}")
    if steps < 1:
        raise ValueError("steps must be positive")
    if replay_user and mode != "online":
        raise ValueError("Replayed users require online rollouts")
    online = mode == "online"
    dataset = ONLINE_TRAIN_DATASET if online else TRAIN_DATASET
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
        ac2_agent=SalesEmailAgent(model=model) if online else None,
        ac2_env=SalesEmailEnvironment() if online else None,
        ac2_user=(ReplaySalesEmailUser() if replay_user else SalesEmailUser()) if online else None,
        ac2_grader=EmDashGrader() if online else None,
        ac2_eval_dataset=EVAL_DATASET if online else None,
        judge_model=token_judge if online else None,
        extra_train_args={"rmsd_hinter_path": HINTER_PATH} if online else {},
    )


CONFIGS = {mode: training_config(mode) for mode in ("offline", "one_step", "online")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit an OPSD run; all modes allocate training compute.")
    parser.add_argument("--project", default="sales-email-opsd")
    parser.add_argument("--mode", choices=CONFIGS, default="online")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="One training step over at most ten tasks")
    parser.add_argument("--replay-user", action="store_true", help="Replay recorded rep turns online")
    parser.add_argument(
        "--no-token-judge", action="store_true", help="Train on every response token in online mode"
    )
    args = parser.parse_args()
    config = training_config(
        args.mode,
        model=args.model,
        steps=args.steps,
        smoke=args.smoke,
        replay_user=args.replay_user,
        token_judge=None if args.no_token_judge else JUDGE_MODEL,
    )
    print(Client(project=args.project).train.run(config))


if __name__ == "__main__":
    main()
