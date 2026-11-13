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
EVAL_TASKS = ("fix-git", "openssl-selfsigned-cert", "regex-log")


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

    tasks = sorted(asyncio.run(load_tasks()), key=lambda task: task.id)
    client = Client(project=args.project)
    try:
        client.datasets.get(args.dataset)
    except DatasetNotFoundError:
        client.datasets.create(
            args.dataset,
            description="Terminal-Bench 2.0 tasks run through Harbor by a self-hosted Harness API.",
        )
    first = [task for task in tasks if task.id in EVAL_TASKS]
    client.datasets.add_tasks(args.dataset, tasks=first)
    client.datasets.add_tasks(
        args.dataset, tasks=[task for task in tasks if task.id not in EVAL_TASKS]
    )
    print(
        f"Uploaded {len(tasks)} {HARBOR_DATASET} tasks to {args.project}/{args.dataset}"
    )


if __name__ == "__main__":
    main()
