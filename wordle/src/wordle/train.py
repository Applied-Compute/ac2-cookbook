"""Submit a small Modal run and stop it before the wall-clock budget expires."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from ac2.sdk import Client, TrainingConfig

from .dataset import PROJECT, TRAIN_DATASET

CONFIG = TrainingConfig(
    model="Qwen/Qwen3-4B",
    n_training_replicas=4,
    n_inference_replicas=4,
    num_train_steps=10,
    samples_per_problem=4,
    problem_batch_size=8,
    ac2_agent="WordleAgent",
    ac2_env="WordleEnvironment",
    ac2_grader="WordleGrader",
    ac2_train_dataset=TRAIN_DATASET,
    training_agent_names=["WordleAgent"],
    backend="modal",
    gpu_type="b200",
    eval_mode="off",
    eval_before_train=False,
    max_response_len=4096,
    max_total_len=8192,
    rollout_sample_timeout=180,
    global_sampling_concurrency=32,
    keep_last_checkpoints=1,
    extra_train_args={"save_interval": 5},
    name="wordle-qwen3-4b-modal-smoke",
)

TERMINAL_STATUSES = {"completed", "succeeded", "failed", "cancelled", "canceled", "stopped"}


def monitor(client: Client, train_id: str, deadline: float) -> str:
    """Keep the launcher alive; stop on deadline, errors, or interruption."""
    terminal = False
    last_status = None
    try:
        while time.monotonic() < deadline:
            run = client.train.get(train_id)
            status = (run.status or "pending").lower()
            if status != last_status:
                print(f"{train_id}: {status}", flush=True)
                last_status = status
            if status in TERMINAL_STATUSES:
                terminal = True
                return status
            time.sleep(min(15, max(0, deadline - time.monotonic())))
        print("Wall-clock limit reached; stopping training and attached evals.", flush=True)
        return "deadline"
    finally:
        if not terminal:
            client.train.stop(train_id, stop_evals=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-minutes", type=float, default=55,
                        help="Submission-to-stop budget, at most 55 minutes (leaves shutdown time).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_minutes <= 55:
        parser.error("--max-minutes must be greater than zero and at most 55")
    if args.dry_run:
        print(json.dumps(asdict(CONFIG), indent=2))
        return
    if os.environ.get("AC2_MODE") != "prod":
        raise SystemExit("Set AC2_MODE=prod before launching training.")
    client = Client(project=PROJECT)
    deadline = time.monotonic() + args.max_minutes * 60
    run = client.train.run(CONFIG)
    print(f"Training ID: {run.train_id}", flush=True)
    try:
        output = Path("runs") / run.train_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "submission.json").write_text(json.dumps({
            "train_id": run.train_id, "project": PROJECT,
            "config": asdict(CONFIG), "max_minutes": args.max_minutes,
        }, indent=2) + "\n")
    except BaseException:
        client.train.stop(run.train_id, stop_evals=True)
        raise
    status = monitor(client, run.train_id, deadline)
    print(f"Final status: {status}", flush=True)
    if status not in {"completed", "succeeded"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
