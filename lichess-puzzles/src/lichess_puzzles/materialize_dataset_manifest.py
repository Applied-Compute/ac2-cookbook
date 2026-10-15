"""Materialize an exact AC2 dataset snapshot as a local full-task manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ac2.runtime import Task
from ac2.sdk import Client


def _refs_hash(refs: list[Any]) -> str:
    """Return the canonical hash used in AC2 task-ref manifest names."""

    payload = "".join(
        json.dumps(
            {"id": ref.id, "blob_uri": ref.blob_uri},
            separators=(",", ":"),
        )
        + "\n"
        for ref in refs
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_task_ids(path: Path) -> list[str]:
    with path.open("rb") as raw:
        is_gzip = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    task_ids: list[str] = []
    with opener(path, "rt", encoding="utf-8") as manifest:
        for line in manifest:
            if not line.strip():
                continue
            task = Task.model_validate_json(line)
            if task.id is None:
                raise ValueError(f"Task in {path} has no registered id.")
            task_ids.append(task.id)
    return task_ids


def materialize_dataset_manifest(
    *,
    dataset: str,
    output: Path,
    expected_count: int,
    expected_refs_hash: str,
    project: str | None = None,
    client: Client | None = None,
) -> None:
    """Fetch one verified dataset snapshot through AC2 and write it atomically."""

    if client is None:
        if project is None:
            raise ValueError("project is required when no AC2 client is supplied")
        client = Client(project=project)
    refs = client.datasets.active_tasks(dataset)
    actual_refs_hash = _refs_hash(refs)
    if actual_refs_hash != expected_refs_hash:
        raise ValueError(
            f"Dataset {dataset!r} task-ref hash changed: expected "
            f"{expected_refs_hash}, got {actual_refs_hash}."
        )
    if len(refs) != expected_count:
        raise ValueError(
            f"Dataset {dataset!r} has {len(refs)} tasks; expected {expected_count}."
        )

    expected_ids = [ref.id for ref in refs]
    if output.is_file() and _validated_task_ids(output) == expected_ids:
        print(
            f"Validated existing local AC2 manifest with {expected_count} tasks: {output}"
        )
        return

    tasks_by_id = {task.id: task for task in client.datasets.load_tasks(dataset)}
    if set(tasks_by_id) != set(expected_ids):
        missing = len(set(expected_ids) - set(tasks_by_id))
        extra = len(set(tasks_by_id) - set(expected_ids))
        raise ValueError(
            f"Dataset content did not match registered refs: missing={missing}, extra={extra}."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as raw:
            temporary_path = Path(raw.name)
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                for task_id in expected_ids:
                    payload = tasks_by_id[task_id].model_dump(mode="json")
                    compressed.write(
                        (
                            json.dumps(payload, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        ).encode("utf-8")
                    )
        if _validated_task_ids(temporary_path) != expected_ids:
            raise ValueError("Materialized manifest failed its post-write validation.")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Materialized local AC2 manifest with {expected_count} tasks: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-refs-hash", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    materialize_dataset_manifest(
        project=args.project,
        dataset=args.dataset,
        output=args.output,
        expected_count=args.expected_count,
        expected_refs_hash=args.expected_refs_hash,
    )


if __name__ == "__main__":
    main()
