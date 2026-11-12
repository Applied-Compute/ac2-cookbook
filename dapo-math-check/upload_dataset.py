from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError

PROJECT = "dapo-math-check"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATASETS = {
    "dapo-math-check-train2048": "dapo-mid-train2048.parquet",
    "dapo-math-check-eval256": "dapo-mid-eval256.parquet",
}
OPSD_DATASET = "dapo-math-check-opsd-train"
OPSD_TRANSCRIPTS = "opsd-transcripts.jsonl"


def row_to_task(row: pd.Series) -> Task:
    answer = str(row["reward_model"]["ground_truth"])
    question = row["prompt"]
    task_id = hashlib.sha256(f"{question}|{answer}".encode()).hexdigest()[:16]
    return Task(
        id=task_id,
        input=[Message(role="user", content=question)],
        env_params={"answer": answer},
        grader_params={"answer": answer},
    )


def transcript_tasks() -> list[Task]:
    lines = (DATA_DIR / OPSD_TRANSCRIPTS).read_text().splitlines()
    return [Task.model_validate_json(line) for line in lines]


def upload_one(client: Client, *, dataset: str, tasks: list[Task]) -> None:
    try:
        client.datasets.get(dataset)
    except DatasetNotFoundError:
        client.datasets.create(dataset)
    print(f"[ac2] {dataset}: uploading {len(tasks)} tasks, this may take a minute")
    client.datasets.add_tasks(dataset, tasks=tasks)
    print(f"[ac2] {dataset}: {len(tasks)} tasks")


def main() -> None:
    client = Client(project=PROJECT)
    for dataset, filename in DATASETS.items():
        df = pd.read_parquet(DATA_DIR / filename)
        upload_one(client, dataset=dataset, tasks=[row_to_task(row) for _, row in df.iterrows()])
    upload_one(client, dataset=OPSD_DATASET, tasks=transcript_tasks())


if __name__ == "__main__":
    main()
