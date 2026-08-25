from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig, TrainingCustomHarnessConfig

from .relay import running_relay_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="swebench-opencode")
    parser.add_argument("--dataset", default="swebench-opencode-smoke")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--samples-per-problem", type=int, default=2)
    parser.add_argument("--problem-batch-size", type=int, default=1)
    args = parser.parse_args()

    sampling_concurrency = args.samples_per_problem * args.problem_batch_size
    client = Client(project=args.project)
    config = TrainingConfig(
        model=args.model,
        n_training_replicas=1,
        n_inference_replicas=1,
        num_train_steps=args.steps,
        keep_last_checkpoints=3,
        samples_per_problem=args.samples_per_problem,
        problem_batch_size=args.problem_batch_size,
        ac2_orchestrator="SwebenchOpenCodeOrchestrator",
        ac2_grader="SwebenchVerifiedGrader",
        ac2_train_dataset=args.dataset,
        custom_harness=TrainingCustomHarnessConfig(
            relay_deployment_id=running_relay_id(client),
        ),
        training_agent_names=["opencode"],
        eval_before_train=False,
        eval_mode="off",
        max_response_len=8_192,
        max_total_len=16_384,
        global_sampling_concurrency=sampling_concurrency,
        rollout_sample_timeout=3_600,
        name="swebench-opencode-smoke",
    )
    client.train.run(config)


if __name__ == "__main__":
    main()
