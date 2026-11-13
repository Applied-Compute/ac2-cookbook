from pydantic import BaseModel, ConfigDict


class HarnessOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
