from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig

from .agent import DOMAIN_ENVIRONMENT, Tau2Agent
from .dataloader import DOMAINS, Domain
from .graders import Tau2BenchGrader
from .user import Tau2BenchDefaultUser


def train_config(domain: Domain = "airline") -> TrainingConfig:
    return TrainingConfig(
        model="Qwen/Qwen3-4B",
        n_training_replicas=4,
        n_inference_replicas=4,
        samples_per_problem=4,
        problem_batch_size=8,
        num_train_steps=20,
        ac2_agent=Tau2Agent(domain=domain),
        ac2_env=DOMAIN_ENVIRONMENT[domain](),
        ac2_grader=Tau2BenchGrader(),
        ac2_user=Tau2BenchDefaultUser(),
        ac2_train_dataset=f"tau2bench-{domain}-train",
        ac2_eval_dataset=f"tau2bench-{domain}-test",
    )


CONFIG = train_config()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="airline")
    args = parser.parse_args()
    Client(project="tau2bench").train.run(train_config(args.domain))


if __name__ == "__main__":
    main()
