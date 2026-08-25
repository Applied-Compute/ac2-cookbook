"""Load tau2bench tasks and per-domain policies.

Files are fetched and cached lazily on first use under
``dataloader/cache/{domain}/``.

The source data defines three task splits:

- ``train``: tasks reserved for training rollouts;
- ``test``: tasks reserved for held-out evaluation;
- ``base``: legacy subset.

``load_tau2_tasks(domain, split=...)`` filters the full ``tasks.json`` list to
the requested split. ``load_tau2_split_ids`` is exposed so callers can
introspect the splits without loading tasks.

Each task is mapped to an ac2 :class:`~ac2.runtime.Task`:

- ``user_scenario``        → markdown ``user_instructions`` for ``Tau2BenchUser``
- ``initial_state``        → ``env_params`` (env applies it on setup) + ``grader_params``
- ``evaluation_criteria``  → flattened into ``grader_params``

If a task specifies ``reward_basis``, it is preserved; otherwise the grader
uses its default (``[DB, COMMUNICATE]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

from ac2.runtime import Message, Task

DOMAINS = ("airline", "retail", "telecom")
SPLITS = ("train", "test", "base")
Split = Literal["train", "test", "base"]

TAU2_RAW_BASE = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains"
POLICY_FILE = {"airline": "policy.md", "retail": "policy.md", "telecom": "main_policy.md"}
DB_FILE = {"airline": "db.json", "retail": "db.json", "telecom": "db.toml"}
SPLIT_TASKS_FILE = "split_tasks.json"
CACHE_DIR = Path(__file__).parent / "cache"


def _fetch(url: str, dest: Path) -> bytes:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url) as resp:
            dest.write_bytes(resp.read())
    return dest.read_bytes()


def load_tau2_policy(domain: str) -> str:
    fname = POLICY_FILE[domain]
    return _fetch(f"{TAU2_RAW_BASE}/{domain}/{fname}", CACHE_DIR / domain / fname).decode()


def load_tau2_db(domain: str) -> Path:
    """Lazily fetch the domain DB into the cache and return its path."""
    fname = DB_FILE[domain]
    dest = CACHE_DIR / domain / fname
    _fetch(f"{TAU2_RAW_BASE}/{domain}/{fname}", dest)
    return dest


def load_tau2_split_ids(domain: str, split: Split) -> list[str]:
    """Return the task IDs in ``split`` for ``domain``."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS!r}.")
    raw = json.loads(
        _fetch(
            f"{TAU2_RAW_BASE}/{domain}/{SPLIT_TASKS_FILE}",
            CACHE_DIR / domain / SPLIT_TASKS_FILE,
        )
    )
    if split not in raw:
        raise KeyError(
            f"split_tasks.json for {domain!r} has no {split!r} key; "
            f"saw {sorted(raw.keys())!r}."
        )
    return list(raw[split])


def _format_user_instructions(user_scenario: dict[str, Any]) -> str:
    instructions = user_scenario.get("instructions", {})
    parts: list[str] = []
    if isinstance(instructions, str):
        parts.append(instructions)
    else:
        if instructions.get("reason_for_call"):
            parts.append(f"## Reason for contacting support\n{instructions['reason_for_call']}")
        if instructions.get("known_info"):
            parts.append(f"## Information you know\n{instructions['known_info']}")
        if instructions.get("unknown_info"):
            parts.append(f"## Information you don't know\n{instructions['unknown_info']}")
        if instructions.get("task_instructions"):
            parts.append(f"## Special instructions\n{instructions['task_instructions']}")
    if user_scenario.get("persona"):
        parts.append(f"## Your persona\n{user_scenario['persona']}")
    return "\n\n".join(parts)


def load_tau2_tasks(
    domain: str,
    *,
    split: Split | None = None,
    num_tasks: int | None = None,
) -> list[Task]:
    """Load tau2-bench tasks for ``domain``, optionally filtered to ``split``.

    When ``split`` is supplied, only tasks whose ID appears in
    ``split_tasks.json[split]`` are returned, preserving the split order
    listed in that file (not the order in ``tasks.json``). Missing IDs raise
    ``KeyError`` so split drift is visible immediately.
    """
    raw = json.loads(
        _fetch(
            f"{TAU2_RAW_BASE}/{domain}/tasks.json",
            CACHE_DIR / domain / "tasks.json",
        )
    )
    by_id = {t["id"]: t for t in raw}

    if split is not None:
        ids = load_tau2_split_ids(domain, split)
        missing = [tid for tid in ids if tid not in by_id]
        if missing:
            raise KeyError(
                f"split_tasks.json[{split!r}] references task IDs {missing!r} not present in tasks.json."
            )
        selected = [by_id[tid] for tid in ids]
    else:
        selected = list(raw)

    if num_tasks is not None:
        selected = selected[:num_tasks]

    tasks: list[Task] = []
    for t in selected:
        ec = t.get("evaluation_criteria") or {}
        initial_state = t.get("initial_state")
        grader_params: dict[str, Any] = {
            "domain": domain,
            "user_instructions": _format_user_instructions(t.get("user_scenario") or {}),
            "expected_actions": ec.get("actions") or [],
            "communicate_info": ec.get("communicate_info") or [],
            "env_assertions": ec.get("env_assertions") or [],
            "nl_assertions": ec.get("nl_assertions") or [],
            "initial_state": initial_state,
        }
        if ec.get("reward_basis"):
            grader_params["reward_basis"] = ec["reward_basis"]
        env_params: dict[str, Any] = {}
        if initial_state:
            env_params["initial_state"] = initial_state

        tasks.append(
            Task(
                id=f"{domain}-{t['id']}",
                input=[Message(role="user", content="(customer simulator drives the conversation)")],
                env_params=env_params,
                grader_params=grader_params,
            )
        )
    return tasks
