"""Submit the standard Lichess-puzzles GRPO training run to AC2 Dispatch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict

from ac2.sdk import Client, TrainingConfig

from lichess_puzzles.train import (
    DEFAULT_CLUSTER,
    DEFAULT_EVAL_DATASET,
    DEFAULT_GPU_TYPE,
    DEFAULT_IMAGE,
    DEFAULT_TRAIN_DATASET,
    DEFAULT_TRAIN_TASKS,
    build_config,
)

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_IPB = 16
DEFAULT_SPI = 8
DEFAULT_STEPS = 500


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standard, Dispatch-only GRPO launch configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-dataset", default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--train-dataset-size", type=int, default=DEFAULT_TRAIN_TASKS)
    parser.add_argument("--eval-dataset", default=DEFAULT_EVAL_DATASET)
    parser.add_argument("--problem-batch-size", type=int, default=DEFAULT_IPB, help="IPB")
    parser.add_argument("--samples-per-problem", type=int, default=DEFAULT_SPI, help="SPI")
    parser.add_argument(
        "--global-sampling-concurrency",
        type=int,
        help="Defaults to IPB x SPI so every rollout sample can be scheduled.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=DEFAULT_STEPS,
        help="500 steps draws all 8,000 prompts once at the default IPB of 16.",
    )
    parser.add_argument("--n-training-replicas", type=int, default=1)
    parser.add_argument("--n-inference-replicas", type=int, default=4)
    parser.add_argument("--max-response-len", type=int, default=8_192)
    parser.add_argument("--rollout-sample-timeout", type=int)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-samples-per-problem", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--keep-last-checkpoints", type=int, default=5)
    parser.add_argument("--checkpoint-save-local-concurrency", type=int, default=1)
    parser.add_argument("--checkpoint-mode", choices=["inference", "resumable"], default="inference")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--gpu-type", choices=["h100", "b200", "b300"], default=DEFAULT_GPU_TYPE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--priority", choices=["low", "medium", "high", "critical"], default="medium")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--wandb-project",
        help="Defaults to the selected AC2 project when Weights & Biases logging is enabled.",
    )
    parser.add_argument("--name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact managed-project config without contacting AC2.",
    )
    args = parser.parse_args(argv)
    if args.global_sampling_concurrency is None:
        args.global_sampling_concurrency = args.problem_batch_size * args.samples_per_problem
    if args.name is None:
        model_name = args.model.rsplit("/", maxsplit=1)[-1].lower().replace(".", "")
        args.name = (
            f"lichess-{model_name}-grpo-ipb{args.problem_batch_size}-"
            f"spi{args.samples_per_problem}-steps{args.num_train_steps}"
        )
    if args.wandb_project is None:
        args.wandb_project = args.project

    # build_config owns the shared environment contract. These fields select its
    # Dispatch path without exposing alternate backend launch modes here.
    args.backend = "dispatch"
    args.eval_mode = "sidecar"
    args.tool_call_parser = None
    args.reasoning_parser = None
    return args


def print_plan(args: argparse.Namespace, config: TrainingConfig) -> None:
    """Print geometry and the fully resolved managed-project configuration."""
    samples_per_step = args.problem_batch_size * args.samples_per_problem
    prompt_draws = args.problem_batch_size * args.num_train_steps
    if args.train_dataset_size <= 0:
        raise ValueError("train dataset size must be positive")
    print(
        f"backend=dispatch cluster={config.cluster_id} gpu={config.gpu_type} "
        f"topology=train:{config.n_training_replicas}/inference:{config.n_inference_replicas}"
    )
    print(
        f"IPB={args.problem_batch_size} SPI={args.samples_per_problem} "
        f"samples_per_step={samples_per_step} "
        f"global_sampling_concurrency={config.global_sampling_concurrency}"
    )
    print(
        f"steps={args.num_train_steps} prompt_draws={prompt_draws} "
        f"nominal_dataset_passes={prompt_draws / args.train_dataset_size:.3f}"
    )
    print(
        f"eval_mode={config.eval_mode} eval_interval={config.eval_interval} "
        f"eval_dataset={config.ac2_eval_dataset} save_interval={args.save_interval}"
    )
    print("project_code=managed project version uploaded by Client.train.run")
    print(json.dumps(asdict(config), indent=2, sort_keys=True, default=str))


def submit(args: argparse.Namespace, config: TrainingConfig) -> object:
    """Create the managed training run through AC2 Dispatch."""
    return Client(project=args.project).train.run(config)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    print_plan(args, config)
    if args.dry_run:
        print("dry run only; no AC2 training job was submitted")
        return
    run = submit(args, config)
    print(f"train_id={run.train_id}")
    if getattr(run, "dispatch_job_id", None):
        print(f"dispatch_job_id={run.dispatch_job_id}")


if __name__ == "__main__":
    main()
