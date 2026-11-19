"""Print or submit a bounded remote evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass

from ac2.sdk import Client, EvalConfig, EvalServe
from ac2.sdk.task_sources import DatasetSource

from .agent import (
    DEFAULT_EVAL_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    QWEN36_MAX_OUTPUT_TOKENS,
    QWEN36_MODEL,
)
from .chess_logic import OBSERVATION_FORMAT
from .grader import PROGRESS_REWARD_WEIGHT, REWARD_SCHEME
from .tool_contract import (
    ACTION_INTERFACE,
    DEFAULT_TRIES,
    QWEN36_REASONING_PARSER,
    QWEN36_TOOL_CALL_PARSER,
)

DEFAULT_CLUSTER: str | None = None
DEFAULT_DATASET = "lichess-puzzles-eval-v1"
QWEN36_SERVE_GPUS = 2


@dataclass(frozen=True)
class EvalVariant:
    """One fully static managed-project policy selection."""

    model: str
    orchestrator: str
    max_output_tokens: int
    reasoning_mode: str
    serve: EvalServe | None = None


EVAL_VARIANTS = {
    "luna": EvalVariant(
        model=DEFAULT_EVAL_MODEL,
        orchestrator="LichessPuzzleOrchestrator",
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        reasoning_mode="low",
    ),
    "qwen36": EvalVariant(
        model=QWEN36_MODEL,
        orchestrator="LichessPuzzleQwen36Orchestrator",
        max_output_tokens=QWEN36_MAX_OUTPUT_TOKENS,
        reasoning_mode="thinking-default",
        serve=EvalServe(
            model=QWEN36_MODEL,
            num_gpus=QWEN36_SERVE_GPUS,
            args={
                "reasoning-parser": QWEN36_REASONING_PARSER,
                "tool-call-parser": QWEN36_TOOL_CALL_PARSER,
            },
        ),
    ),
}


def build_config(args: argparse.Namespace) -> EvalConfig:
    variant = EVAL_VARIANTS[args.variant]
    if variant.serve is not None and args.backend != "dispatch":
        raise ValueError("Self-served eval variants require the dispatch backend")
    dataset: str | DatasetSource = args.dataset
    if args.num_tasks is not None:
        dataset = DatasetSource(dataset=args.dataset, num_tasks=args.num_tasks)
    cluster_id = args.cluster
    return EvalConfig(
        orchestrator=variant.orchestrator,
        grader="LichessPuzzleGrader",
        dataset=dataset,
        cluster_id=cluster_id,
        backend="dispatch",
        max_parallel=args.max_parallel,
        num_samples=args.num_samples,
        priority=args.priority,
        name=(
            args.name
            or f"lichess-puzzles {variant.model} {variant.max_output_tokens}tok {args.dataset}"
        ),
        description=(
            "Normalized Lichess prefix-progress reward with a completion bonus. Binary solve and "
            "failure categories remain in grader artifacts; a final completion at the configured "
            "output ceiling indicates budget exhaustion."
        ),
        tags={
            "max_output_tokens": variant.max_output_tokens,
            "model": variant.model,
            "action_interface": ACTION_INTERFACE,
            "environment_default_tries": DEFAULT_TRIES,
            "observation_format": OBSERVATION_FORMAT,
            "reasoning_mode": variant.reasoning_mode,
            "reasoning_parser": (
                variant.serve.args.get("reasoning-parser", "") if variant.serve else ""
            ),
            "tool_call_parser": (
                variant.serve.args.get("tool-call-parser", "")
                if variant.serve
                else "provider_native"
            ),
            "reward_scheme": REWARD_SCHEME,
            "progress_reward_weight": PROGRESS_REWARD_WEIGHT,
            "serve_num_gpus": variant.serve.num_gpus if variant.serve else 0,
            "task": "lichess-puzzles",
            "variant": args.variant,
        },
        serve=variant.serve,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--variant", choices=sorted(EVAL_VARIANTS), default="luna")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=32)
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--priority", choices=["low", "medium", "high", "critical"], default="low")
    parser.add_argument("--name")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    args.backend = "dispatch"
    return args


async def submit(args: argparse.Namespace, config: EvalConfig) -> None:
    run = await Client(project=args.project).eval.run(config)
    print(f"eval_id={run.eval_id}")
    if getattr(run, "dispatch_job_id", None):
        print(f"dispatch_job_id={run.dispatch_job_id}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    print(json.dumps(asdict(config), indent=2, sort_keys=True, default=str))
    if not args.submit:
        print("dry run only; pass --submit to launch")
        return
    asyncio.run(submit(args, config))


if __name__ == "__main__":
    main()
