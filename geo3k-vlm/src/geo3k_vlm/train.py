import argparse
from ac2.sdk import TrainingConfig
from .settings import client
from .tasks import TRAIN_DATASET, EVAL_DATASET


def config(cluster_id):
    return TrainingConfig(
        model="Qwen/Qwen3.5-9B-VL",
        n_training_replicas=1,
        n_inference_replicas=1,
        num_train_steps=20,
        samples_per_problem=4,
        problem_batch_size=8,
        ac2_agent="Geo3KAgent",
        ac2_env="Geo3KEnvironment",
        ac2_grader="Geo3KGrader",
        training_agent_names=["Geo3KAgent"],
        ac2_train_dataset=TRAIN_DATASET,
        ac2_eval_dataset=EVAL_DATASET,
        max_response_len=4096,
        eval_interval=10,
        eval_samples_per_problem=1,
        backend="dispatch",
        cluster_id=cluster_id,
        name="geo3k-vlm",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", required=True)
    args = parser.parse_args()
    run = client().train.run(config(args.cluster))
    print(run.train_id)
