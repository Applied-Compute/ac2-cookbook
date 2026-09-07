"""Submit a small Modal run and stop it before the wall-clock budget expires."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from ac2.sdk import Client, TrainingConfig

from .dataset import EVAL_DATASET, PROJECT, TRAIN_DATASET

CONFIG = TrainingConfig(
    model="Qwen/Qwen3-4B",
    n_training_replicas=4,
    n_inference_replicas=4,
    num_train_steps=40,
    samples_per_problem=4,
    problem_batch_size=8,
    ac2_agent="WordleAgent",
    ac2_env="WordleEnvironment",
    ac2_grader="WordleGrader",
    ac2_user="WordleUser",
    ac2_train_dataset=TRAIN_DATASET,
    ac2_eval_dataset=EVAL_DATASET,
    training_agent_names=["WordleAgent"],
    backend="modal",
    gpu_type="b200",
    eval_mode="sidecar",
    eval_before_train=True,
    eval_interval=5,
    eval_samples_per_problem=1,
    n_eval_replicas=1,
    eval_global_sampling_concurrency=32,
    # Training replaces the agent's completion client, so its eval kwargs do
    # not apply here. This budget covers the entire game, including tool turns.
    apply_chat_template_kwargs={"enable_thinking": True},
    max_response_len=24576,
    max_total_len=32768,
    rollout_sample_timeout=600,
    global_sampling_concurrency=32,
    # Eval sidecars may still be queued when later checkpoints are saved.
    keep_last_checkpoints=8,
    # Keep each training replica on one B200. Automatic context parallelism
    # otherwise multiplies GPU allocation when the episode budget increases.
    extra_train_args={"save_interval": 5, "context_parallel_size": 1},
    name="wordle-qwen3-4b-modal-40steps-eval5",
)

TERMINAL_STATUSES = {"completed", "succeeded", "failed", "cancelled", "canceled", "stopped"}
SUCCESS_STATUSES = {"completed", "succeeded"}


def monitor(client: Client, train_id: str, deadline: float, *, expected_evals: int = 0) -> str:
    """Keep training and attached evals within the same wall-clock budget."""
    terminal = False
    last_status = None
    last_evals = None
    try:
        while time.monotonic() < deadline:
            run = client.train.get(train_id)
            status = (run.status or "pending").lower()
            if status != last_status:
                print(f"{train_id}: {status}", flush=True)
                last_status = status
            if status in TERMINAL_STATUSES:
                if not expected_evals:
                    terminal = True
                    return status
                if status not in SUCCESS_STATUSES:
                    return status  # Stop any attached evals in finally.
                children = list(client.jobs.list(parent_job_id=run.job_id, limit=100))
                eval_states = {job.id: job.state.lower() for job in children}
                if eval_states != last_evals:
                    print(f"Attached evals: {json.dumps(eval_states, sort_keys=True)}", flush=True)
                    last_evals = eval_states
                if all(state in TERMINAL_STATUSES for state in eval_states.values()):
                    terminal = True
                    if len(eval_states) != expected_evals:
                        print(f"Expected {expected_evals} evals, found {len(eval_states)}.", flush=True)
                        return "eval_missing"
                    return status if all(s in SUCCESS_STATUSES for s in eval_states.values()) else "eval_failed"
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
    expected_evals = 1 + (CONFIG.num_train_steps + CONFIG.eval_interval - 1) // CONFIG.eval_interval
    status = monitor(client, run.train_id, deadline, expected_evals=expected_evals)
    print(f"Final status: {status}", flush=True)
    if status not in {"completed", "succeeded"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
