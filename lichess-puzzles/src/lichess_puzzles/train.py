"""Shared configuration for the public Lichess-puzzles GRPO launcher."""

from __future__ import annotations

import argparse
import json

from ac2.sdk import TrainingConfig

from .chess_logic import OBSERVATION_FORMAT
from .grader import PROGRESS_REWARD_WEIGHT, REWARD_SCHEME
from .tool_contract import (
    ACTION_INTERFACE,
    DEFAULT_TRIES,
    training_reasoning_parser,
    training_tool_call_parser,
)

DEFAULT_CLUSTER: str | None = None
DEFAULT_BACKEND = "dispatch"
DEFAULT_GPU_TYPE = "b300"
DEFAULT_IMAGE = "appliedcomp/mantis:dev"
DEFAULT_TRAIN_DATASET = "lichess-puzzles-train-v1"
DEFAULT_EVAL_DATASET = "lichess-puzzles-eval-v1"
DEFAULT_TRAIN_TASKS = 8_000
MEGATRON_SAVE_LOCAL_CONCURRENCY_ENV = "MEGATRON_SAVE_LOCAL_CONCURRENCY"


def build_config(args: argparse.Namespace) -> TrainingConfig:
    samples_per_step = args.problem_batch_size * args.samples_per_problem
    if args.global_sampling_concurrency < samples_per_step:
        raise ValueError(
            "global sampling concurrency must be at least IPB x SPI "
            f"({args.problem_batch_size} x {args.samples_per_problem} = {samples_per_step})"
        )
    if args.checkpoint_save_local_concurrency < 0:
        raise ValueError("checkpoint save local concurrency must be nonnegative")
    tool_call_parser = args.tool_call_parser or training_tool_call_parser(args.model)
    reasoning_parser = args.reasoning_parser or training_reasoning_parser(args.model)
    sidecar_enabled = args.eval_mode == "sidecar"
    cluster_id = args.cluster
    extra_train_args: dict[str, str | int | bool] = {
        "tool_call_parser": tool_call_parser,
        "save_interval": args.save_interval,
    }
    if reasoning_parser is not None:
        # cp-server maps this passthrough key to --sglang-reasoning-parser.
        extra_train_args["sglang_reasoning_parser"] = reasoning_parser
    if args.checkpoint_save_local_concurrency > 0:
        # Mantis parses this JSON and injects the values into every Ray training actor.
        # Megatron's local save gate is implemented only in AsyncRequest.execute_sync,
        # so explicitly disable Mantis's default async torch_dist worker as well.
        extra_train_args["no_async_save"] = True
        extra_train_args["train_env_vars"] = json.dumps(
            {MEGATRON_SAVE_LOCAL_CONCURRENCY_ENV: str(args.checkpoint_save_local_concurrency)},
            sort_keys=True,
        )
    if args.checkpoint_mode == "inference":
        extra_train_args.update(no_save_optim=True, no_save_rng=True)
    return TrainingConfig(
        model=args.model,
        n_training_replicas=args.n_training_replicas,
        n_inference_replicas=args.n_inference_replicas,
        samples_per_problem=args.samples_per_problem,
        problem_batch_size=args.problem_batch_size,
        num_train_steps=args.num_train_steps,
        ac2_orchestrator="LichessPuzzleOrchestrator",
        ac2_grader="LichessPuzzleGrader",
        training_agent_names=["LichessPuzzleAgent"],
        ac2_train_dataset=args.train_dataset,
        ac2_eval_dataset=args.eval_dataset if sidecar_enabled else None,
        max_response_len=args.max_response_len,
        rollout_sample_timeout=args.rollout_sample_timeout,
        eval_mode=args.eval_mode,
        eval_before_train=sidecar_enabled,
        eval_interval=args.eval_interval if sidecar_enabled else None,
        eval_samples_per_problem=args.eval_samples_per_problem,
        keep_last_checkpoints=args.keep_last_checkpoints,
        global_sampling_concurrency=args.global_sampling_concurrency,
        wandb_enabled=args.wandb,
        wandb_project=getattr(args, "wandb_project", None) if args.wandb else None,
        cluster_id=cluster_id,
        backend=DEFAULT_BACKEND,
        gpu_type=args.gpu_type,
        image=args.image,
        code_branches=None,
        priority=args.priority,
        name=args.name or f"lichess-puzzles {args.model}",
        tags={
            "action_interface": ACTION_INTERFACE,
            "environment_default_tries": DEFAULT_TRIES,
            "observation_format": OBSERVATION_FORMAT,
            "ipb": args.problem_batch_size,
            "spi": args.samples_per_problem,
            "samples_per_step": samples_per_step,
            "backend": args.backend,
            "eval_mode": args.eval_mode,
            "sidecar_evaluations": sidecar_enabled,
            "max_response_len": args.max_response_len,
            "keep_last_checkpoints": args.keep_last_checkpoints,
            "save_interval": args.save_interval,
            "checkpoint_save_local_concurrency": args.checkpoint_save_local_concurrency,
            "checkpoint_async_save": args.checkpoint_save_local_concurrency == 0,
            "checkpoint_mode": args.checkpoint_mode,
            "reward_scheme": REWARD_SCHEME,
            "progress_reward_weight": PROGRESS_REWARD_WEIGHT,
            "tool_call_parser": tool_call_parser,
            "reasoning_parser": reasoning_parser or "",
        },
        extra_train_args=extra_train_args,
    )
