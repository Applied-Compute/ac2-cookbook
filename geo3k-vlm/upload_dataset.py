import argparse

from ac2.sdk import DatasetNotFoundError
from datasets import load_dataset
from geo3k_vlm.settings import client
from geo3k_vlm.tasks import SOURCE, SOURCE_REVISION, TRAIN_DATASET, EVAL_DATASET, row_to_task


def main(revision: str):
    data = load_dataset(SOURCE, revision=revision)
    api = client()
    for split, name in (("train", TRAIN_DATASET), ("validation", EVAL_DATASET)):
        tasks = [row_to_task(row) for row in data[split]]
        try:
            api.datasets.get(name)
        except DatasetNotFoundError:
            api.datasets.create(name)
        api.datasets.add_tasks(name, tasks=tasks)
        print(f"{name}: {len(tasks)} tasks from {SOURCE}@{revision}/{split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=SOURCE_REVISION, help="Hugging Face dataset commit SHA")
    main(parser.parse_args().revision)
