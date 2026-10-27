"""Build every dataset locally, without API keys or model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sales_email_opsd.dashes import POLICY_VERSION
from sales_email_opsd.data import content_hash, load_conversations, write_jsonl
from sales_email_opsd.dataset import build_tasks

DATA_DIR = Path(__file__).resolve().parent / "data"


def build(data_dir: Path, max_boundaries: int = 3) -> None:
    source = data_dir / "traces.jsonl"
    records = load_conversations(source)
    tasks = build_tasks(records, max_boundaries=max_boundaries)
    outputs = {
        "tasks_one_step.jsonl": tasks.one_step,
        "tasks_online.jsonl": tasks.online,
        "tasks_eval.jsonl": tasks.evaluation,
    }
    for filename, rows in outputs.items():
        write_jsonl(data_dir / filename, rows)
        print(f"{filename}: {len(rows)} tasks")
    manifest = {
        "schema_version": 2,
        "policy_version": POLICY_VERSION,
        "source_sha256": content_hash(source.read_text(encoding="utf-8")),
        "max_boundaries": max_boundaries,
        "outputs": {
            name: {"count": len(rows), "sha256": content_hash((data_dir / name).read_text(encoding="utf-8"))}
            for name, rows in outputs.items()
        },
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-boundaries", type=int, default=3)
    args = parser.parse_args()
    build(args.data_dir, args.max_boundaries)
