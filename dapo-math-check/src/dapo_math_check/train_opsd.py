from __future__ import annotations

import argparse

from ac2.sdk import Client, TrainingConfig

_COMMON = dict(model="Qwen/Qwen3-4B", n_training_replicas=4, samples_per_problem=1, problem_batch_size=8)

CONFIGS = {
    "offline": TrainingConfig(
        **_COMMON,
        method="opsd",
        opsd_rollout_mode="offline",
        n_inference_replicas=0,
        ac2_train_dataset="dapo-math-check-opsd-train",
    ),
    "one_step": TrainingConfig(
        **_COMMON,
        method="opsd",
        opsd_rollout_mode="one_step",
        n_inference_replicas=4,
        ac2_train_dataset="dapo-math-check-opsd-train",
    ),
    "online": TrainingConfig(
        **_COMMON,
        method="opsd",
        opsd_rollout_mode="online",
        n_inference_replicas=4,
        ac2_agent="DapoMathAgent",
        ac2_env="DapoMathCheckEnvironment",
        ac2_grader="DapoMathCheckGrader",
        training_agent_names=["DapoMathAgent"],
        ac2_train_dataset="dapo-math-check-opsd-online-train",
        ac2_eval_dataset="dapo-math-check-eval256",
        judge_model="gpt-5-mini",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=CONFIGS, default="offline")
    args = parser.parse_args()
    Client(project="dapo-math-check").train.run(CONFIGS[args.mode])


if __name__ == "__main__":
    main()
