from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig, TrainingCustomHarnessConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="tau2bench")
    parser.add_argument("--dataset", default="tau2bench-airline-train")
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
            samples_per_problem=4,
            problem_batch_size=2,
            global_sampling_concurrency=8,
            ac2_train_dataset=args.dataset,
            custom_harness=TrainingCustomHarnessConfig(),
            eval_mode="off",
            name="tau2bench",
        )
    )
    print(run.train_id)


if __name__ == "__main__":
    main()
