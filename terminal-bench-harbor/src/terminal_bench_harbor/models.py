from pydantic import BaseModel, ConfigDict


class HarborDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ref: str = "latest"


class CodingRolloutOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str | None = None
    trial_id: str | None = None
    trial_uri: str | None = None
    rewards: dict[str, float | int] | None = None
    error: str | None = None
