from __future__ import annotations

import argparse
import asyncio

from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError
from harbor.models.job.config import DatasetConfig
from harbor.models.task.task import Task as HarborTask
from harbor.tasks.client import TaskClient

HARBOR_DATASET = "terminal-bench/terminal-bench-2"
HARBOR_REF = "sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b"
SMOKE_TASKS = ("fix-git", "openssl-selfsigned-cert", "regex-log")


async def load_tasks() -> list[Task]:
    dataset = DatasetConfig(name=HARBOR_DATASET, ref=HARBOR_REF)
    task_ids = [config.get_task_id() for config in await dataset.get_task_configs()]
    downloads = await TaskClient().download_tasks(task_ids)
    return [
        Task(
            id=task_id.get_name().split("/")[-1],
            input=[Message(role="user", content=HarborTask(path).instruction)],
            env_params={
                "name": HARBOR_DATASET,
                "ref": HARBOR_REF,
                "task_name": task_id.get_name(),
            },
        )
        for task_id, path in zip(task_ids, downloads.paths, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="terminal-bench-harbor")
    parser.add_argument("--dataset", default="terminal-bench")
    args = parser.parse_args()

    tasks = asyncio.run(load_tasks())
    smoke_tasks = [task for task in tasks if task.id in SMOKE_TASKS]
    client = Client(project=args.project)
    for dataset, dataset_tasks in (
        (args.dataset, tasks),
        (f"{args.dataset}-smoke", smoke_tasks),
    ):
        try:
            client.datasets.get(dataset)
        except DatasetNotFoundError:
            client.datasets.create(
                dataset,
                description="Terminal-Bench 2.0 tasks run through Harbor by a self-hosted Harness API.",
            )
        client.datasets.add_tasks(dataset, tasks=dataset_tasks)
        print(f"Uploaded {len(dataset_tasks)} tasks to {args.project}/{dataset}")


if __name__ == "__main__":
    main()
