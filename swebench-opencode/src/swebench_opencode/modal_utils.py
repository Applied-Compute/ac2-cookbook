from __future__ import annotations

from dataclasses import dataclass

import modal


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


async def run_command(
    sandbox: modal.Sandbox,
    command: str,
    *,
    env: dict[str, str | None] | None = None,
    timeout_seconds: int = 0,
) -> CommandResult:
    proc = await sandbox.exec.aio(
        "bash",
        "-c",
        command,
        env=env,
        timeout=timeout_seconds or None,
        text=False,
    )
    await proc.wait.aio()
    stdout = (await proc.stdout.read.aio()).decode("utf-8", errors="replace")
    stderr = (await proc.stderr.read.aio()).decode("utf-8", errors="replace")
    if proc.returncode is None:
        raise RuntimeError("Sandbox process ended without an exit code")
    return CommandResult(returncode=proc.returncode, stdout=stdout, stderr=stderr)
