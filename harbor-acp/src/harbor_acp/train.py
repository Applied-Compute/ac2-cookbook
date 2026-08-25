from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig, TrainingCustomHarnessConfig

from .relay import running_relay_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="harbor-acp")
    parser.add_argument("--dataset", default="harbor-hello-world")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    client = Client(project=args.project)
    config = TrainingConfig(
        model=args.model,
        n_training_replicas=1,
        n_inference_replicas=1,
        num_train_steps=1,
        keep_last_checkpoints=1,
        samples_per_problem=2,
        problem_batch_size=1,
        ac2_orchestrator="HarborACPOrchestrator",
        ac2_grader="HarborRewardGrader",
        ac2_train_dataset=args.dataset,
        custom_harness=TrainingCustomHarnessConfig(
            relay_deployment_id=running_relay_id(client),
        ),
        training_agent_names=["opencode"],
        eval_before_train=False,
        eval_mode="off",
        max_response_len=4_096,
        max_total_len=8_192,
        global_sampling_concurrency=2,
        rollout_sample_timeout=1_800,
        name="harbor-acp-hello-world-smoke",
    )
    client.train.run(config)


if __name__ == "__main__":
    main()
