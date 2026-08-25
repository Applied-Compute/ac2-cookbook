from __future__ import annotations

import argparse

from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError

HARBOR_DATASET = "harbor/hello-world"
HARBOR_REF = "latest"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="byoh-modal-coding-api")
    parser.add_argument("--dataset", default="byoh-modal-coding-eval")
    args = parser.parse_args()

    client = Client(project=args.project)
    try:
        client.datasets.get(args.dataset)
    except DatasetNotFoundError:
        client.datasets.create(
            args.dataset,
            description="Harbor coding task executed by a self-hosted Harness API.",
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
                env_params={"name": HARBOR_DATASET, "ref": HARBOR_REF},
            )
        ],
    )
    print(f"Uploaded {HARBOR_DATASET}@{HARBOR_REF} to {args.project}/{args.dataset}")


if __name__ == "__main__":
    main()
