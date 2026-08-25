from __future__ import annotations

import tempfile
from typing import cast

from pydantic import BaseModel, JsonValue
from swebench.harness.constants import SWEbenchInstance
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

from ac2.runtime import EnvironmentProtocol, Grader, GraderOutput, Trace

from .environment import SwebenchOpenCodeEnvironment
from .modal_utils import run_command


class SwebenchGraderParams(BaseModel):
    datapoint: dict[str, JsonValue]


class SwebenchVerifiedGrader(Grader):
    async def _grade(
        self,
        grader_params: dict[str, JsonValue] | None,
        trace: Trace,
        env: EnvironmentProtocol,
    ) -> GraderOutput:
        if not isinstance(env, SwebenchOpenCodeEnvironment):
            raise TypeError("SwebenchVerifiedGrader requires SwebenchOpenCodeEnvironment")

        params = SwebenchGraderParams.model_validate(grader_params or {})
        test_spec = make_test_spec(
            cast(SWEbenchInstance, params.datapoint),
            namespace="swebench",
        )
        await env.write_file("/root/eval.sh", test_spec.eval_script)
        result = await run_command(
            env.sandbox,
            "cd /testbed && bash /root/eval.sh",
            timeout_seconds=env.grader_timeout,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as test_log:
            test_log.write(result.stdout)
            if result.stderr:
                test_log.write(f"\n{result.stderr}")
            test_log.flush()
            report = get_eval_report(
                test_spec=test_spec,
                prediction={
                    "instance_id": test_spec.instance_id,
                    "model_patch": "",
                },
                test_log_path=test_log.name,
                include_tests_status=True,
            )

        instance_report = report.get(test_spec.instance_id, {})
        resolved = bool(instance_report.get("resolved", False))
        tests_status = instance_report.get("tests_status", {})
        fail_to_pass = tests_status.get("FAIL_TO_PASS", {})
        pass_to_pass = tests_status.get("PASS_TO_PASS", {})
        fail_to_pass_succeeded = sorted(fail_to_pass.get("success", []))
        fail_to_pass_failed = sorted(fail_to_pass.get("failure", []))
        pass_to_pass_succeeded = sorted(pass_to_pass.get("success", []))
        pass_to_pass_failed = sorted(pass_to_pass.get("failure", []))

        return GraderOutput(
            score=1.0 if resolved else 0.0,
            reasoning=(
                f"resolved={resolved}; "
                f"fail-to-pass={len(fail_to_pass_succeeded)}/"
                f"{len(fail_to_pass_succeeded) + len(fail_to_pass_failed)}; "
                f"pass-to-pass={len(pass_to_pass_succeeded)}/"
                f"{len(pass_to_pass_succeeded) + len(pass_to_pass_failed)}"
            ),
            artifacts={
                "resolved": resolved,
                "fail_to_pass_succeeded": fail_to_pass_succeeded,
                "fail_to_pass_failed": fail_to_pass_failed,
                "pass_to_pass_succeeded": pass_to_pass_succeeded,
                "pass_to_pass_failed": pass_to_pass_failed,
                "grader_returncode": result.returncode,
            },
        )
