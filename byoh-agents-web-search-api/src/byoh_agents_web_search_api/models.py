from pydantic import BaseModel, ConfigDict, Field


class WebSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[str] = Field(default_factory=list)
    error: str | None = None
