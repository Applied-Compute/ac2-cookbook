from ac2.runtime import EnvironmentProtocol, Grader, GraderOutput, Trace

from .environment import DapoMathCheckEnvironment


class DapoMathCheckGrader(Grader):
    async def _grade(
        self,
        grader_params: dict | None,
        trace: Trace,
        env: EnvironmentProtocol,
    ) -> GraderOutput:
        if not isinstance(env, DapoMathCheckEnvironment):
            raise TypeError(f"DapoMathCheckGrader requires DapoMathCheckEnvironment, got {type(env).__name__}.")
        terminated_reason = env.terminated_reason
        terminated_correct = env.terminated_correct
        expected = str((grader_params or {}).get("answer", ""))
        score = 1.0 if terminated_correct else 0.0
        status = "PASS" if terminated_correct else "FAIL"
        return GraderOutput(
            score=score,
            reasoning=(
                f"[{status}] Expected: {expected} | "
                f"terminated_reason={terminated_reason!r} | "
                f"terminated_correct={terminated_correct}"
            ),
            artifacts={
                "expected": expected,
                "terminated_reason": terminated_reason,
            },
        )
