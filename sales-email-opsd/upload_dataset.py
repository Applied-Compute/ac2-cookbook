"""Build the tasks from the bundled conversations and register them as AC2 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ac2.runtime import Task
from ac2.sdk import Client, DatasetNotFoundError

from sales_email_opsd.data import load_conversations
from sales_email_opsd.dataset import EVAL_DATASET, ONLINE_TRAIN_DATASET, TRAIN_DATASET, build_tasks

TRACES = Path(__file__).resolve().parent / "data" / "traces.jsonl"


def upload(client: Client, datasets: dict[str, list[Task]]) -> None:
    existing: dict[str, set[str]] = {}
    missing = []
    # Inspect every dataset before changing any membership.
    for dataset, tasks in datasets.items():
        try:
            client.datasets.get(dataset)
        except DatasetNotFoundError:
            missing.append(dataset)
            existing[dataset] = set()
            continue
        hashes = {task.content_hash for task in client.datasets.load_tasks(dataset)}
        if hashes - {task.content_hash for task in tasks}:
            raise ValueError(f"{dataset} contains a different snapshot; choose new names in dataset.py")
        existing[dataset] = hashes
    for dataset in missing:
        client.datasets.create(dataset)
    for dataset, tasks in datasets.items():
        additions = [task for task in tasks if task.content_hash not in existing[dataset]]
        if additions:
            client.datasets.add_tasks(dataset, tasks=additions)
        print(f"{dataset}: {len(tasks)} tasks ({len(additions)} added)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="sales-email-opsd")
    args = parser.parse_args()
    tasks = build_tasks(load_conversations(TRACES))
    datasets = {
        TRAIN_DATASET: tasks.one_step,
        ONLINE_TRAIN_DATASET: tasks.online,
        EVAL_DATASET: tasks.evaluation,
    }
    upload(Client(project=args.project), datasets)


if __name__ == "__main__":
    main()
