from __future__ import annotations

from ac2.runtime import Environment
from harbor.models.trial.result import TrialResult
from pydantic import BaseModel, JsonValue


class HarborDataset(BaseModel):
    name: str
    ref: str = "latest"


class HarborEnvironment(Environment):
    def __init__(self) -> None:
        self.dataset: HarborDataset | None = None
        self.trial_result: TrialResult | None = None

    async def setup(self, env_params: dict[str, JsonValue]) -> None:
        self.dataset = HarborDataset.model_validate(env_params)

    def require_dataset(self) -> HarborDataset:
        if self.dataset is None:
            raise RuntimeError("HarborEnvironment has not been set up")
        return self.dataset

    def require_trial_result(self) -> TrialResult:
        if self.trial_result is None:
            raise RuntimeError("Harbor has not produced a trial result")
        return self.trial_result
