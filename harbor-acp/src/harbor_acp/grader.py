from __future__ import annotations

from ac2.runtime import EnvironmentProtocol, Grader, GraderOutput, Trace
from pydantic import JsonValue

from .environment import HarborEnvironment


class HarborRewardGrader(Grader):
    async def _grade(
        self,
        grader_params: dict[str, JsonValue] | None,
        trace: Trace,
        env: EnvironmentProtocol,
    ) -> GraderOutput:
        if not isinstance(env, HarborEnvironment):
            raise TypeError("HarborRewardGrader requires HarborEnvironment")

        trial = env.require_trial_result()
        if trial.exception_info is not None:
            return GraderOutput(
                score=0,
                reasoning=f"Harbor trial failed: {trial.exception_info.exception_message}",
                artifacts={"exception_type": trial.exception_info.exception_type},
            )
        if trial.verifier_result is None or trial.verifier_result.rewards is None:
            raise RuntimeError("Harbor trial did not produce verifier rewards")

        rewards = trial.verifier_result.rewards
        reward = rewards.get("reward")
        if not isinstance(reward, int | float):
            raise TypeError("Harbor verifier did not produce a numeric reward")
        return GraderOutput(
            score=float(reward),
            reasoning=f"Harbor verifier reward={reward}",
            artifacts={"rewards": rewards, "trial_uri": trial.trial_uri},
        )
