from __future__ import annotations

from ac2.sdk import Client, TrainingConfig


CONFIG = TrainingConfig(
    model="Qwen/Qwen3-4B",
    n_training_replicas=4,
    n_inference_replicas=4,
    num_train_steps=3,
    samples_per_problem=4,
    problem_batch_size=8,
    ac2_agent="DapoMathAgent",
    ac2_env="DapoMathCheckEnvironment",
    ac2_grader="DapoMathCheckGrader",
    ac2_train_dataset="dapo-math-check-train2048",
    ac2_eval_dataset="dapo-math-check-eval256",
    training_agent_names=["DapoMathAgent"],
)


def main() -> None:
    Client(project="dapo-math-check").train.run(CONFIG)


if __name__ == "__main__":
    main()
