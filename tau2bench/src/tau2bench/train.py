from __future__ import annotations

from ac2.sdk import Client, TrainingConfig


CONFIG = TrainingConfig(
    model="Qwen/Qwen3-4B",
    n_training_replicas=4,
    n_inference_replicas=4,
    samples_per_problem=4,
    problem_batch_size=8,
    num_train_steps=20,
    ac2_agent="Tau2AirlineAgent",
    ac2_env="AirlineEnvironment",
    ac2_grader="Tau2BenchGrader",
    ac2_user="Tau2BenchDefaultUser",
    ac2_train_dataset="tau2bench-airline-train",
    ac2_eval_dataset="tau2bench-airline-test",
    training_agent_names=["Tau2AirlineAgent"],
)


def main() -> None:
    Client(project="tau2bench").train.run(CONFIG)


if __name__ == "__main__":
    main()
