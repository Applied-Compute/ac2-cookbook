from __future__ import annotations

import argparse

from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError

DEFAULT_DATASET = "harbor-hello-world"
TASK_NAME = "harbor/hello-world"
TASK_REF = "latest"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="harbor-acp")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    client = Client(project=args.project)
    try:
        client.datasets.get(args.dataset)
    except DatasetNotFoundError:
        client.datasets.create(
            args.dataset,
            description="Harbor hello-world task run through an ACP agent.",
        )
    client.datasets.add_tasks(
        args.dataset,
        tasks=[
            Task(
                id="harbor-hello-world",
                input=[
                    Message(
                        role="user",
                        content='Create hello.txt containing exactly "Hello, world!".',
                    )
                ],
                env_params={"name": TASK_NAME, "ref": TASK_REF},
            )
        ],
    )
    print(f"Uploaded {TASK_NAME}@{TASK_REF} to {args.project}/{args.dataset}")


if __name__ == "__main__":
    main()
