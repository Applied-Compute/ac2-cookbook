from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json

from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError
from datasets import load_dataset

SOURCE = "hiyouga/geometry3k"
SOURCE_REVISION = "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"


def row_to_task(row: dict) -> Task:
    content = [{"type": "text", "text": row["problem"].replace("<image>", "").strip()}]
    for image in row["images"]:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            }
        )
    if len(content) == 1:
        raise ValueError("Geometry3K tasks require an image")
    answer = str(row["answer"])
    task_id = hashlib.sha256(
        json.dumps([content, answer], sort_keys=True).encode()
    ).hexdigest()[:24]
    return Task(
        id=task_id,
        input=[Message(role="user", content=content)],
        grader_params={"answer": answer},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="geo3k-vlm")
    args = parser.parse_args()

    client = Client(project=args.project)
    data = load_dataset(SOURCE, revision=SOURCE_REVISION)
    for split, name in (("train", "geo3k-train"), ("validation", "geo3k-validation")):
        tasks = [row_to_task(row) for row in data[split]]
        try:
            client.datasets.get(name)
        except DatasetNotFoundError:
            client.datasets.create(name)
        client.datasets.add_tasks(name, tasks=tasks)
        print(f"Uploaded {len(tasks)} tasks to {args.project}/{name}")


if __name__ == "__main__":
    main()
