from __future__ import annotations

from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError

PROJECT = "rock-paper-scissors"
DATASET = "rock-paper-scissors"
DEFAULT_PROMPT = "Play rock-paper-scissors against the mystery opponent. Begin:"


def build_tasks() -> list[Task]:
    return [
        Task(
            input=[Message(role="user", content=DEFAULT_PROMPT)],
            env_params={"max_rounds": 40},
        )
    ]


def main() -> None:
    client = Client(project=PROJECT)
    try:
        client.datasets.get(DATASET)
    except DatasetNotFoundError:
        client.datasets.create(DATASET)
        print(f"Created dataset {DATASET!r}.")

    tasks = build_tasks()
    client.datasets.add_tasks(dataset=DATASET, tasks=tasks)
    print(f"Uploaded {len(tasks)} task(s) to {DATASET!r}.")


if __name__ == "__main__":
    main()
