"""Submit a saved Lichess-puzzles checkpoint evaluation to AC2 Dispatch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from ac2.sdk import Client

from lichess_puzzles.train import DEFAULT_CLUSTER


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one managed checkpoint-evaluation request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--train-id", required=True, help="Source AC2 train_xxx identifier.")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Saved distributed checkpoint directory, normally an s3:// URI.",
    )
    parser.add_argument(
        "--hf-checkpoint",
        required=True,
        help="Base Hugging Face checkpoint used for conversion, normally an s3:// URI.",
    )
    parser.add_argument("--iter-num", required=True, type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument("--name")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--rollout-num-gpus", type=int, default=8)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--cpu")
    parser.add_argument("--convert-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--priority", choices=["low", "medium", "high", "critical"], default="medium")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Schedule the checkpoint sidecar after printing the request.",
    )
    args = parser.parse_args(argv)
    if args.iter_num < 0:
        raise ValueError("iter-num must be nonnegative")
    if args.rollout_num_gpus < 1 or args.rollout_num_gpus_per_engine < 1:
        raise ValueError("rollout GPU counts must be positive")
    if args.experiment_name is None:
        args.experiment_name = f"{args.train_id}-checkpoint-{args.iter_num:07d}"
    return args


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    """Build the AC2 SDK request without inferring storage paths or image details."""
    return {
        "train_job_id": args.train_id,
        "experiment_name": args.experiment_name,
        "checkpoint_dir": args.checkpoint_dir,
        "hf_checkpoint": args.hf_checkpoint,
        "iter_num": args.iter_num,
        "rollout_num_gpus": args.rollout_num_gpus,
        "rollout_num_gpus_per_engine": args.rollout_num_gpus_per_engine,
        "sglang_mem_fraction_static": args.sglang_mem_fraction_static,
        "cpu": args.cpu,
        "convert_checkpoint": args.convert_checkpoint,
        "cluster_id": args.cluster,
        "backend": "dispatch",
        "name": args.name,
        "priority": args.priority,
    }


def submit(args: argparse.Namespace) -> object:
    """Schedule the service-owned Dispatch checkpoint-evaluation sidecar."""
    checkpoint_eval = getattr(Client(project=args.project).eval, "run_checkpoint", None)
    if checkpoint_eval is None:
        raise RuntimeError(
            "Checkpoint evaluation requires an AC2 SDK exposing "
            "Client.eval.run_checkpoint. Run `uv sync --upgrade-package ac2` and retry."
        )
    return checkpoint_eval(**build_request(args))


def main() -> None:
    args = parse_args()
    request = build_request(args)
    print(json.dumps(request, indent=2, sort_keys=True))
    if not args.submit:
        print("dry run only; pass --submit to schedule the Dispatch checkpoint sidecar")
        return
    run = submit(args)
    print(f"eval_id={run.eval_id}")
    if getattr(run, "job_id", None):
        print(f"dispatch_job_id={run.job_id}")
    if getattr(run, "status", None):
        print(f"status={run.status}")


if __name__ == "__main__":
    main()
