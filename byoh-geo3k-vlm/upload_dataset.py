import argparse
import os

from ac2.sdk import Client, DatasetNotFoundError
from datasets import load_dataset

from byoh_geo3k_vlm.tasks import (
    EVAL_DATASET,
    SMOKE_EVAL_DATASET,
    SMOKE_TRAIN_DATASET,
    SOURCE,
    SOURCE_REVISION,
    TRAIN_DATASET,
    GeometryProblem,
    row_to_task,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Upload eight tasks per split to separate smoke datasets")
    args = parser.parse_args()
    client = Client(project=os.environ["AC2_PROJECT"], active_org_id=os.environ["AC2_ORG_ID"])
    names = (SMOKE_TRAIN_DATASET, SMOKE_EVAL_DATASET) if args.smoke else (TRAIN_DATASET, EVAL_DATASET)
    for split, name in zip(("train", "validation"), names, strict=True):
        data = load_dataset(SOURCE, revision=SOURCE_REVISION, split=f"{split}[:8]" if args.smoke else split)
        tasks = [row_to_task(GeometryProblem.model_validate(row)) for row in data]
        try:
            client.datasets.get(name)
        except DatasetNotFoundError:
            client.datasets.create(name)
        client.datasets.add_tasks(name, tasks=tasks)
        print(f"{name}: {len(tasks)} tasks from {SOURCE}@{SOURCE_REVISION}/{split}")


if __name__ == "__main__":
    main()
