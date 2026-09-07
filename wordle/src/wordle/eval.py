import argparse
import asyncio
import os

from ac2.sdk import Client, DatasetSource, EvalConfig, EvalServe

from .dataset import EVAL_DATASET, PROJECT, build_datasets


async def main(*, local: bool = False, model: str | None = None,
               base_url: str | None = None, api_key_env: str | None = None,
               num_tasks: int = 32) -> None:
    if os.environ.get("AC2_MODE") != "prod":
        raise SystemExit("Set AC2_MODE=prod before evaluating.")
    if not local and (base_url or api_key_env):
        raise ValueError("--base-url and --api-key-env are for local evals; remote eval serves its own model.")
    # AC2 reloads project modules while building the version manifest. Configure
    # the agent through env vars so local inference settings survive that reload.
    for key, value in {"WORDLE_EVAL_MODEL": model, "WORDLE_EVAL_BASE_URL": base_url,
                       "WORDLE_EVAL_API_KEY_ENV": api_key_env}.items():
        if local and value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    config = EvalConfig(
        agent="WordleAgent", env="WordleEnvironment", grader="WordleGrader",
        user="WordleUser",
        num_samples=1, max_parallel=4, on_error="raise",
        **({"tasks": build_datasets()[EVAL_DATASET][:num_tasks]} if local else {
            "dataset": DatasetSource(dataset=EVAL_DATASET, num_tasks=num_tasks),
            "serve": EvalServe(
                model=model or "Qwen/Qwen3-4B", num_gpus=1,
                args={"tool-call-parser": "hermes", "reasoning-parser": "qwen3"},
            ),
        }),
    )
    run = await Client(project=PROJECT).eval.run(config, local=local)
    print(run.eval_job_id if local else run.eval_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--model", help="Optional inference model or AC2 endpoint for the smoke eval.")
    parser.add_argument("--base-url", help="OpenAI-compatible inference API URL.")
    parser.add_argument("--api-key-env", help="Environment variable containing the inference API key.")
    parser.add_argument("--num-tasks", type=int, default=32, choices=range(1, 257),
                        metavar="1-256", help="Number of held-out words (default: 32).")
    args = parser.parse_args()
    asyncio.run(main(local=args.local, model=args.model,
                     base_url=args.base_url, api_key_env=args.api_key_env,
                     num_tasks=args.num_tasks))
