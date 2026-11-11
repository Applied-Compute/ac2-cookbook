from pydantic import BaseModel, ConfigDict


class MinimalOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    error: str | None = None
