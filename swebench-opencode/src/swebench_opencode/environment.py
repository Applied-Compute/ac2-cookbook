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
        self._image_head = ""
        self._model_patch_error: str | None = None
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

    @property
    def model_patch_error(self) -> str | None:
        return self._model_patch_error

    async def setup(self, env_params: dict[str, JsonValue]) -> None:
        params = SwebenchTaskParams.model_validate(env_params)
        self._image = params.image
        self._model_patch_error = None
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

        head = await run_command(
            self.harness_sandbox, "git rev-parse HEAD", timeout_seconds=60
        )
        if head.returncode != 0:
            raise RuntimeError(f"Failed to read image commit: {head.stderr}")
        self._image_head = head.stdout.strip()

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
git diff --binary "$IMAGE_HEAD"
""",
            env={"IMAGE_HEAD": self._image_head},
            timeout_seconds=300,
        )
        if patch_result.returncode != 0:
            raise RuntimeError(f"Failed to fetch model patch: {patch_result.stderr}")

        self._grading_sandbox = await self._create_sandbox(
            modal.Image.from_registry(self._image).entrypoint([])
        )

        head_result = await run_command(
            self.sandbox, "git rev-parse HEAD", timeout_seconds=60
        )
        if head_result.returncode != 0:
            raise RuntimeError(
                f"Failed to verify grading sandbox image commit: {head_result.stderr.strip()}"
            )
        if head_result.stdout.strip() != self._image_head:
            raise RuntimeError(
                "Grading sandbox does not match the harness image commit: "
                f"expected {self._image_head}, got {head_result.stdout.strip()}"
            )
        status_result = await run_command(
            self.sandbox,
            "git status --porcelain --untracked-files=no",
            timeout_seconds=60,
        )
        if status_result.returncode != 0:
            raise RuntimeError(
                f"Failed to verify grading sandbox worktree: {status_result.stderr.strip()}"
            )
        if status_result.stdout.strip():
            raise RuntimeError(
                f"Grading sandbox has pre-existing tracked changes: {status_result.stdout.strip()}"
            )

        if patch_result.stdout:
            await self.sandbox.filesystem.write_text.aio(
                patch_result.stdout, "/tmp/model.patch"
            )
            check_result = await run_command(
                self.sandbox,
                "git apply --check --binary /tmp/model.patch",
                timeout_seconds=300,
            )
            if check_result.returncode != 0:
                self._model_patch_error = (
                    check_result.stderr.strip()
                    or f"git apply --check exited with code {check_result.returncode}"
                )
                return patch_result.stdout

            apply_result = await run_command(
                self.sandbox,
                "git apply --binary /tmp/model.patch",
                timeout_seconds=300,
            )
            if apply_result.returncode != 0:
                raise RuntimeError(
                    "Patch passed git apply --check but failed during application: "
                    f"{apply_result.stderr}"
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
