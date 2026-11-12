from ac2.harness.types import RolloutStatus
from ac2.runtime import EnvironmentProtocol, Grader, GraderOutput, Trace
from pydantic import BaseModel, JsonValue

from .models import CodingRolloutOutput


class _RolloutOutput(BaseModel):
    status: RolloutStatus
    outputs: CodingRolloutOutput


class HarborRewardGrader(Grader):
    async def _grade(
        self,
        grader_params: dict[str, JsonValue] | None,
        trace: Trace,
        env: EnvironmentProtocol,
    ) -> GraderOutput:
        raw = (await env.inspect(["rollout_output"])).get("rollout_output")
        result = _RolloutOutput.model_validate(raw).outputs
        if result.error is not None:
            return GraderOutput(
                score=0,
                reasoning=f"Harbor trial failed: {result.error}",
                artifacts={"trial_uri": result.trial_uri or ""},
            )
        if result.rewards is None:
            raise RuntimeError("Harbor trial did not produce verifier rewards")

        reward = result.rewards.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, int | float):
            raise TypeError("Harbor verifier did not produce a numeric reward")
        return GraderOutput(
            score=float(reward),
            reasoning=f"Harbor verifier reward={reward}",
            artifacts={"rewards": result.rewards, "trial_uri": result.trial_uri},
        )
