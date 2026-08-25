from __future__ import annotations

import asyncio

import modal
from pydantic import BaseModel, JsonValue

from ac2.runtime import Environment
from ac2.sdk import blob_download_url

from .modal_utils import run_command

WORKDIR = "/testbed"
OPENCODE_VERSION = "1.15.11"
SANDBOX_TIMEOUT_SECONDS = 3 * 60 * 60


class SwebenchTaskParams(BaseModel):
    image: str
    task_blob: str
    datapoint: dict[str, JsonValue]


class SwebenchOpenCodeEnvironment(Environment):
    workdir = WORKDIR
    grader_timeout = 240

    def __init__(self) -> None:
        self._harness_sandbox: modal.Sandbox | None = None
        self._grading_sandbox: modal.Sandbox | None = None
        self._image = ""
        self._base_commit = ""
        self.datapoint: dict[str, JsonValue] = {}

    @property
    def sandbox(self) -> modal.Sandbox:
        if self._grading_sandbox is None:
            raise RuntimeError("SWE-bench grading sandbox is not ready")
        return self._grading_sandbox

    @property
    def harness_sandbox(self) -> modal.Sandbox:
        if self._harness_sandbox is None:
            raise RuntimeError("OpenCode harness sandbox is not running")
        return self._harness_sandbox

    async def setup(self, env_params: dict[str, JsonValue]) -> None:
        params = SwebenchTaskParams.model_validate(env_params)
        base_commit = params.datapoint.get("base_commit")
        if not isinstance(base_commit, str):
            raise ValueError("datapoint.base_commit is required")

        self._image = params.image
        self._base_commit = base_commit
        self.datapoint = params.datapoint

        opencode_url = (
            f"https://github.com/anomalyco/opencode/releases/download/v{OPENCODE_VERSION}/"
            "opencode-linux-x64-baseline.tar.gz"
        )
        image = (
            modal.Image.from_registry(params.image)
            .entrypoint([])
            .run_commands(
                "mkdir -p /opt/opencode",
                f"curl --fail --silent --show-error --location {opencode_url} "
                "--output /tmp/opencode.tar.gz",
                "tar -xzf /tmp/opencode.tar.gz -C /opt/opencode",
                "chmod +x /opt/opencode/opencode",
            )
        )
        self._harness_sandbox = await self._create_sandbox(image)

        task_grant = await asyncio.to_thread(blob_download_url, params.task_blob)
        result = await run_command(
            self.harness_sandbox,
            """
set -eu
mkdir -p /task
curl --fail --silent --show-error --location "$TASK_BLOB_URL" --output /tmp/task.tar.gz
tar -xzf /tmp/task.tar.gz -C /task
""",
            env={"TASK_BLOB_URL": task_grant.url},
            timeout_seconds=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to prepare harness sandbox: {result.stderr}")

    async def teardown(self) -> None:
        sandboxes = [
            sandbox
            for sandbox in (self._harness_sandbox, self._grading_sandbox)
            if sandbox is not None
        ]
        self._harness_sandbox = None
        self._grading_sandbox = None
        await asyncio.gather(*(sandbox.terminate.aio() for sandbox in sandboxes))

    async def write_file(self, path: str, content: str) -> None:
        await self.sandbox.filesystem.write_text.aio(content, path)

    async def materialize_grading_sandbox(self) -> str:
        patch_result = await run_command(
            self.harness_sandbox,
            """
set -euo pipefail
git add --intent-to-add .
git diff --binary "$BASE_COMMIT"
""",
            env={"BASE_COMMIT": self._base_commit},
            timeout_seconds=300,
        )
        if patch_result.returncode != 0:
            raise RuntimeError(f"Failed to fetch model patch: {patch_result.stderr}")

        self._grading_sandbox = await self._create_sandbox(
            modal.Image.from_registry(self._image).entrypoint([])
        )
        if patch_result.stdout:
            await self.sandbox.filesystem.write_text.aio(
                patch_result.stdout, "/tmp/model.patch"
            )
            apply_result = await run_command(
                self.sandbox,
                "git apply --binary /tmp/model.patch",
                timeout_seconds=300,
            )
            if apply_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to apply model patch: {apply_result.stderr}"
                )
        return patch_result.stdout

    async def _create_sandbox(self, image: modal.Image) -> modal.Sandbox:
        app = await modal.App.lookup.aio(
            "swebench-opencode-custom-harness", create_if_missing=True
        )
        with modal.enable_output():
            return await modal.Sandbox.create.aio(
                "sleep",
                "infinity",
                app=app,
                image=image,
                timeout=SANDBOX_TIMEOUT_SECONDS,
                workdir=self.workdir,
                cpu=4,
                memory=16_384,
            )
