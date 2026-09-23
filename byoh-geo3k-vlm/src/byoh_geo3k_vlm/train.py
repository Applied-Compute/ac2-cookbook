import argparse
import os
from typing import Literal

from ac2.sdk import Client, TrainingConfig, TrainingCustomHarnessConfig

from .grader import Geo3KGrader
from .orchestrator import MAX_RESPONSE_TOKENS, Geo3KByohOrchestrator
from .tasks import EVAL_DATASET, SMOKE_EVAL_DATASET, SMOKE_TRAIN_DATASET, TRAIN_DATASET


def config(cluster_id: str, backend: Literal["dispatch", "modal"] = "dispatch", smoke: bool = False) -> TrainingConfig:
    return TrainingConfig(
        model="Qwen/Qwen3.6-35B-A3B-VL",
        method="grpo",
        n_training_replicas=1,
        n_inference_replicas=1,
        num_train_steps=2 if smoke else 20,
        samples_per_problem=4,
        problem_batch_size=8,
        ac2_orchestrator=Geo3KByohOrchestrator(),
        ac2_grader=Geo3KGrader(),
        custom_harness=TrainingCustomHarnessConfig(),
        ac2_train_dataset=SMOKE_TRAIN_DATASET if smoke else TRAIN_DATASET,
        ac2_eval_dataset=SMOKE_EVAL_DATASET if smoke else EVAL_DATASET,
        max_response_len=MAX_RESPONSE_TOKENS,
        max_total_len=16384,
        eval_before_train=True,
        eval_interval=1 if smoke else 10,
        eval_samples_per_problem=1,
        global_sampling_concurrency=8,
        eval_global_sampling_concurrency=4,
        rollout_sample_timeout=1200,
        keep_last_checkpoints=1,
        backend=backend,
        cluster_id=cluster_id,
        name="byoh-geo3k-vlm-smoke" if smoke else "byoh-geo3k-vlm",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", required=True, help="AC2 cluster ID")
    parser.add_argument("--backend", choices=("dispatch", "modal"), default="dispatch")
    parser.add_argument("--smoke", action="store_true", help="Run two training steps on the smoke datasets")
    args = parser.parse_args()
    client = Client(project=os.environ["AC2_PROJECT"], active_org_id=os.environ["AC2_ORG_ID"])
    run = client.train.run(config(args.cluster, args.backend, args.smoke))
    print(run.train_id)


if __name__ == "__main__":
    main()
