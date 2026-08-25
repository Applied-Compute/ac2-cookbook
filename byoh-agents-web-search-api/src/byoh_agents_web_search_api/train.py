from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig, TrainingCustomHarnessConfig

from .relay import running_relay_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="byoh-agents-web-search-api")
    parser.add_argument("--dataset", default="byoh-web-search-eval")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    client = Client(project=args.project)
    run = client.train.run(
        TrainingConfig(
            model=args.model,
            n_training_replicas=1,
            n_inference_replicas=1,
            num_train_steps=args.steps,
            samples_per_problem=2,
            problem_batch_size=1,
            ac2_grader="WebSearchGrader",
            ac2_train_dataset=args.dataset,
            custom_harness=TrainingCustomHarnessConfig(
                relay_deployment_id=running_relay_id(client),
            ),
            eval_mode="off",
            name="byoh-agents-web-search-api-smoke",
        )
    )
    print(run.train_id)


if __name__ == "__main__":
    main()
