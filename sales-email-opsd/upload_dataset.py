"""Validate the reproducible bundle, then register its versioned AC2 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ac2.runtime import Task
from ac2.sdk import Client, DatasetNotFoundError

from sales_email_opsd.dashes import POLICY_VERSION
from sales_email_opsd.data import content_hash, load_conversations
from sales_email_opsd.dataset import DATASETS, build_tasks

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_bundle(data_dir: Path) -> dict[str, list[Task]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 2 or manifest["policy_version"] != POLICY_VERSION:
        raise ValueError("Dataset policy changed; rerun build_opsd_dataset.py")
    source = data_dir / "traces.jsonl"
    if content_hash(source.read_text(encoding="utf-8")) != manifest["source_sha256"]:
        raise ValueError("Source traces changed; rebuild the datasets")
    tasks = build_tasks(load_conversations(source), max_boundaries=manifest["max_boundaries"])
    expected = dict(zip(DATASETS, (tasks.one_step, tasks.online, tasks.evaluation)))
    bundle = {}
    for filename, dataset in DATASETS.items():
        text = (data_dir / filename).read_text(encoding="utf-8")
        if content_hash(text) != manifest["outputs"][filename]["sha256"]:
            raise ValueError(f"{filename} does not match its manifest")
        rows = [Task.model_validate_json(line) for line in text.splitlines() if line.strip()]
        if rows != expected[filename] or len(rows) != manifest["outputs"][filename]["count"]:
            raise ValueError(f"{filename} does not match the current builder")
        bundle[dataset] = rows
    return bundle


def upload_bundle(client: Client, bundle: dict[str, list[Task]]) -> None:
    existing: dict[str, set[str]] = {}
    missing = []
    # Inspect every dataset before changing any membership.
    for dataset, tasks in bundle.items():
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
    for dataset, tasks in bundle.items():
        additions = [task for task in tasks if task.content_hash not in existing[dataset]]
        if additions:
            client.datasets.add_tasks(dataset, tasks=additions)
        print(f"{dataset}: {len(tasks)} tasks ({len(additions)} added)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="sales-email-opsd")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--check", action="store_true", help="Validate locally without AC2 calls")
    args = parser.parse_args()
    bundle = load_bundle(args.data_dir)
    if args.check:
        for name, tasks in bundle.items():
            print(f"{name}: {len(tasks)} validated tasks")
        return
    upload_bundle(Client(project=args.project), bundle)


if __name__ == "__main__":
    main()
