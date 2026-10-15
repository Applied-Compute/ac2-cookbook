"""Register one-time, immutable prepared task files in the AC2 project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ac2.runtime import Task
from ac2.sdk import Client
from ac2.sdk.errors import DatasetNotFoundError

DEFAULT_MANIFEST = Path("data/prepared/manifest.json")
DEFAULT_BATCH_SIZE = 200
LIST_PAGE_SIZE = 1_000


def load_tasks(path: Path) -> list[Task]:
    tasks: list[Task] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(Task.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid task JSON at {path}:{line_number}") from exc
    return tasks


def existing_puzzle_ids(client: Client, dataset: str) -> set[str]:
    """Read existing membership so an interrupted upload resumes without re-uploading."""

    puzzle_ids: set[str] = set()
    offset = 0
    while True:
        page = client.datasets.list_tasks(
            dataset,
            order="oldest",
            limit=LIST_PAGE_SIZE,
            offset=offset,
        )
        for task in page:
            puzzle_id = task.tags.get("puzzle_id")
            if isinstance(puzzle_id, str):
                puzzle_ids.add(puzzle_id)
        offset += len(page)
        if not page or page.total is None or offset >= page.total:
            break
    return puzzle_ids


def register(
    client: Client,
    *,
    manifest_path: Path,
    splits: set[str] | None,
    allow_existing: bool,
    batch_size: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    datasets = manifest["datasets"]
    for split, dataset in datasets.items():
        if splits is not None and split not in splits:
            continue
        name = str(dataset["name"])
        task_path = manifest_path.parent / str(dataset["file"])
        tasks = load_tasks(task_path)
        expected_count = int(dataset["task_count"])
        if len(tasks) != expected_count:
            raise RuntimeError(f"{task_path} has {len(tasks)} tasks; manifest says {expected_count}")

        try:
            existing = client.datasets.get(name)
        except DatasetNotFoundError:
            existing = None
        if existing is not None and not allow_existing:
            raise RuntimeError(
                f"dataset {name!r} already exists; use a new version or pass --allow-existing "
                "to rely on AC2 content-hash deduplication"
            )
        if existing is None:
            client.datasets.create(
                name,
                description=(
                    "Deterministic Lichess puzzle split, stratified by puzzle rating and "
                    "recorded player-move count. See the project manifest for provenance."
                ),
                tags={
                    "source": "lichess-puzzle-database",
                    "source_license": "CC0-1.0",
                    "split": split,
                    "schema_version": int(manifest["schema_version"]),
                    "source_sha256": str(manifest["source"]["sha256"]),
                },
            )
            print(f"created dataset {name!r}")
        elif existing.task_count == expected_count:
            print(f"dataset {name!r} already has all {expected_count} tasks")
            continue

        existing_ids = existing_puzzle_ids(client, name)
        if existing is not None and len(existing_ids) != existing.task_count:
            raise RuntimeError(
                f"dataset {name!r} has {existing.task_count} tasks but only "
                f"{len(existing_ids)} unique puzzle_id tags; refusing an ambiguous resume"
            )
        remaining = [task for task in tasks if task.tags.get("puzzle_id") not in existing_ids]
        print(
            f"dataset {name!r}: {len(existing_ids)} already registered, "
            f"{len(remaining)} remaining"
        )
        for start in range(0, len(remaining), batch_size):
            stop = min(start + batch_size, len(remaining))
            client.datasets.add_tasks(name, tasks=remaining[start:stop])
            print(f"submitted remaining tasks {start + 1}-{stop} for {name!r}")
        registered = client.datasets.get(name)
        if registered.task_count != expected_count:
            raise RuntimeError(
                f"dataset {name!r} has {registered.task_count} tasks after registration; "
                f"expected {expected_count}. Rerun with --allow-existing to resume safely."
            )
        print(
            f"registered {len(tasks)} prepared tasks in {name!r}; "
            f"AC2 task_count={registered.task_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", action="append")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    register(
        Client(project=args.project),
        manifest_path=args.manifest,
        splits=set(args.split) if args.split else None,
        allow_existing=args.allow_existing,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
