from ac2.harness.types import RolloutStatus
from ac2.runtime import EnvironmentProtocol, Grader, GraderOutput, Trace
from pydantic import BaseModel, ConfigDict, JsonValue

from .models import WebSearchOutput


class _GraderParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_answer: str


class _RolloutOutput(BaseModel):
    status: RolloutStatus
    outputs: WebSearchOutput


class WebSearchGrader(Grader):
    async def _grade(
        self,
        grader_params: dict[str, JsonValue] | None,
        trace: Trace,
        env: EnvironmentProtocol,
    ) -> GraderOutput:
        params = _GraderParams.model_validate(grader_params)
        raw = (await env.inspect(["rollout_output"])).get("rollout_output")
        result = _RolloutOutput.model_validate(raw).outputs
        if result.error is not None:
            return GraderOutput(score=0, reasoning=result.error)

        matched = params.expected_answer.casefold() in result.answer.casefold()
        return GraderOutput(
            score=1 if matched else 0,
            reasoning=f"Expected the answer to contain {params.expected_answer!r}.",
            artifacts={"sources": result.sources},
        )
