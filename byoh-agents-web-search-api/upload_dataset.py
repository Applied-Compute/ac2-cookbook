from __future__ import annotations

import argparse

from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError

TASKS = (
    ("red-planet", "Which planet is commonly called the Red Planet?", "Mars"),
    ("pride-and-prejudice", "Who wrote Pride and Prejudice?", "Jane Austen"),
    ("senegal-capital", "What is the capital of Senegal?", "Dakar"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="byoh-agents-web-search-api")
    parser.add_argument("--dataset", default="byoh-web-search-eval")
    args = parser.parse_args()

    client = Client(project=args.project)
    try:
        client.datasets.get(args.dataset)
    except DatasetNotFoundError:
        client.datasets.create(
            args.dataset, description="Web-search tasks for a self-hosted BYOH harness."
        )
    client.datasets.add_tasks(
        args.dataset,
        tasks=[
            Task(
                id=task_id,
                input=[Message(role="user", content=question)],
                grader_params={"expected_answer": expected_answer},
            )
            for task_id, question, expected_answer in TASKS
        ],
    )
    print(f"Uploaded {len(TASKS)} tasks to {args.project}/{args.dataset}")


if __name__ == "__main__":
    main()
