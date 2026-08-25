from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, JsonValue
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.test_spec.test_spec import make_test_spec

from ac2.runtime import Message, Task
from ac2.sdk import Client, DatasetNotFoundError

DEFAULT_DATASET = "swebench-opencode-smoke"
SWE_BENCH_DATASET = "princeton-nlp/SWE-bench_Verified"


class SwebenchDatapoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str

    def payload(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")


def build_task(client: Client, datapoint: SwebenchDatapoint) -> Task:
    payload = datapoint.payload()
    test_spec = make_test_spec(
        cast(SWEbenchInstance, payload),
        namespace="swebench",
    )
    instruction = (
        "Solve this SWE-bench issue in /testbed. Make the smallest correct change, "
        "do not modify tests, and run the relevant tests.\n\n"
        f"{datapoint.problem_statement}\n"
    )
    with TemporaryDirectory() as directory:
        task_directory = Path(directory) / "task"
        task_directory.mkdir()
        (task_directory / "instruction.md").write_text(instruction)
        task_blob = client.blobs.upload(task_directory)

    return Task(
        id=datapoint.instance_id,
        input=[
            Message(
                role="user",
                content="Read /task/instruction.md and solve the task in /testbed.",
            )
        ],
        env_params={
            "image": test_spec.instance_image_key,
            "task_blob": task_blob.id,
            "datapoint": payload,
        },
        grader_params={"datapoint": payload},
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
        SwebenchDatapoint.model_validate(row)
        for row in rows
        if not wanted_ids or row["instance_id"] in wanted_ids
    ][: args.num_tasks]
    if not datapoints:
        parser.error("No SWE-bench Verified tasks matched")

    client = Client(project=args.project)
    print(f"Preparing {len(datapoints)} task(s), this may take a minute.")
    tasks = [build_task(client, datapoint) for datapoint in datapoints]
    try:
        client.datasets.get(args.dataset)
    except DatasetNotFoundError:
        client.datasets.create(
            args.dataset,
            description="SWE-bench Verified tasks for the OpenCode custom harness.",
        )
    print(f"Uploading {len(tasks)} task(s), this may take a minute.")
    client.datasets.add_tasks(args.dataset, tasks=tasks)
    print(f"Uploaded {len(tasks)} task(s) to {args.project}/{args.dataset}")


if __name__ == "__main__":
    main()
