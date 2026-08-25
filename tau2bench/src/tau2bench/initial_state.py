"""Apply a tau2-bench ``initial_state`` to an env.

Used by domain envs in ``setup`` to seed the predicted DB, and by the grader
when it spins up a fresh gold env to replay expected actions on.
"""

from __future__ import annotations

import inspect
from typing import Any

from ac2.runtime import EnvironmentProtocol


async def apply_initial_state(env: EnvironmentProtocol, initial_state: dict[str, Any] | None) -> None:
    if not initial_state:
        return
    for action in initial_state.get("initialization_actions") or []:
        method = getattr(env, action["func_name"], None)
        if method is None:
            continue
        try:
            result = method(**action.get("arguments", {}))
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass
