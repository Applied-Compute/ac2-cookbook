"""Upload tau2bench tasks to AC2 as one dataset per split.

Run before eval/train to materialize datasets on the server.

By default, registers three datasets per domain:

- ``tau2bench-<domain>-train``
- ``tau2bench-<domain>-test``
- ``tau2bench-<domain>-base``
"""

from __future__ import annotations

import argparse

from ac2.sdk import Client, DatasetNotFoundError

from tau2bench.dataloader import DOMAINS, SPLITS, load_tau2_tasks

PROJECT = "tau2bench"


def _upload_one(client: Client, *, domain: str, split: str, dataset: str, num_tasks: int | None) -> None:
    try:
        client.datasets.get(dataset)
        print(f"Dataset {dataset!r} already exists; appending tasks.")
    except DatasetNotFoundError:
        client.datasets.create(dataset)
        print(f"Created dataset {dataset!r}.")

    tasks = load_tau2_tasks(domain, split=split, num_tasks=num_tasks)
    print(f"Uploading {len(tasks)} task(s) to {dataset!r}, this may take a minute.")
    client.datasets.add_tasks(dataset=dataset, tasks=tasks)
    print(f"Uploaded {len(tasks)} task(s) to {dataset!r} (split={split!r}).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS, default="airline")
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help="Upload only one split. Default: upload all of train, test, base.",
    )
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the dataset name. Requires --split. Default: tau2bench-<domain>-<split>.",
    )
    args = parser.parse_args()

    if args.dataset is not None and args.split is None:
        parser.error("--dataset requires --split to disambiguate which split to upload.")

    splits = (args.split,) if args.split is not None else SPLITS

    client = Client(project=PROJECT)
    for split in splits:
        dataset = args.dataset if args.dataset is not None else f"tau2bench-{args.domain}-{split}"
        _upload_one(client, domain=args.domain, split=split, dataset=dataset, num_tasks=args.num_tasks)


if __name__ == "__main__":
    main()
