from __future__ import annotations

import argparse

from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError
from datasets import load_dataset
from pydantic import JsonValue

DEFAULT_DATASET = "swebench-opencode-smoke"
SWE_BENCH_DATASET = "princeton-nlp/SWE-bench_Verified"


def build_task(datapoint: dict[str, JsonValue]) -> Task:
    return Task(
        id=datapoint["instance_id"],
        input=[
            Message(
                role="user",
                content="Read /task/instruction.md and solve the task in /testbed.",
            )
        ],
        env_params={"datapoint": datapoint},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="swebench-opencode")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--instance-id", action="append", default=[])
    args = parser.parse_args()

    if args.num_tasks < 1:
        parser.error("--num-tasks must be positive")

    rows = load_dataset(SWE_BENCH_DATASET, split="test")
    wanted_ids = set(args.instance_id)
    datapoints = [
        dict(row) for row in rows if not wanted_ids or row["instance_id"] in wanted_ids
    ][: args.num_tasks]
    if not datapoints:
        parser.error("No SWE-bench Verified tasks matched")

    client = Client(project=args.project)
    tasks = [build_task(datapoint) for datapoint in datapoints]
    try:
        client.datasets.get(args.dataset)
    except DatasetNotFoundError:
        client.datasets.create(
            args.dataset,
            description="SWE-bench Verified tasks for the self-hosted OpenCode harness.",
        )
    print(f"Uploading {len(tasks)} task(s), this may take a minute.")
    client.datasets.add_tasks(args.dataset, tasks=tasks)
    print(f"Uploaded {len(tasks)} task(s) to {args.project}/{args.dataset}")


if __name__ == "__main__":
    main()
