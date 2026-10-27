from __future__ import annotations

import argparse
import asyncio

from ac2.sdk import Client, DatasetSource, EvalConfig, ModelConfig

from .agent import BASE_MODEL, SalesEmailAgent
from .dashes import POLICY_VERSION
from .dataset import DATASETS
from .environment import SalesEmailEnvironment
from .grader import EmDashGrader
from .user import ReplaySalesEmailUser, SalesEmailUser


def eval_config(
    *,
    model: str | None = None,
    checkpoint: str | None = None,
    api_model: str | None = None,
    num_tasks: int | None = None,
    simulate_user: bool = False,
) -> EvalConfig:
    if sum(value is not None for value in (model, checkpoint, api_model)) > 1:
        raise ValueError("Choose one of model, checkpoint, or api_model")
    if num_tasks is not None and num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    if checkpoint is not None:
        served_model = ModelConfig(checkpoint_id=checkpoint)
    else:
        served_model = model or (BASE_MODEL if api_model is None else None)
    return EvalConfig(
        agent=SalesEmailAgent(model=api_model or BASE_MODEL),
        env=SalesEmailEnvironment(),
        grader=EmDashGrader(),
        user=SalesEmailUser() if simulate_user else ReplaySalesEmailUser(),
        dataset=DatasetSource(dataset=DATASETS["tasks_eval.jsonl"], num_tasks=num_tasks),
        model=served_model,
        temperature=0.0 if api_model is None else None,
        # Matches training; Qwen3 often thinks past 2048 tokens before it writes the email.
        max_response_len=8192 if api_model is None else None,
        num_samples=1,
        max_parallel=4,
        on_error="raise",
        tags={
            "user_policy": "simulated" if simulate_user else "replay",
            "policy_version": POLICY_VERSION,
        },
    )


CONFIG = eval_config()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the base model, a checkpoint, or a provider model."
    )
    parser.add_argument("--project", default="sales-email-opsd")
    parser.add_argument("--local", action="store_true")
    model = parser.add_mutually_exclusive_group()
    model.add_argument("--model", help="AC2 catalog model (default: the training base model)")
    model.add_argument("--checkpoint", help="AC2 checkpoint ID from the training run")
    model.add_argument("--api-model", help="Provider model or deployed endpoint for API-based evaluation")
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--simulate-user", action="store_true")
    args = parser.parse_args()
    if args.local and not args.api_model:
        parser.error("--local requires --api-model; catalog models and checkpoints run remotely")
    config = eval_config(
        model=args.model,
        checkpoint=args.checkpoint,
        api_model=args.api_model,
        num_tasks=args.num_tasks,
        simulate_user=args.simulate_user,
    )
    print(await Client(project=args.project).eval.run(config, local=args.local))


if __name__ == "__main__":
    asyncio.run(main())
