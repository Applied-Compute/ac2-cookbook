import base64
import hashlib
import io
import json

from ac2.runtime import Message, Task
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, JsonValue

SOURCE = "hiyouga/geometry3k"
SOURCE_REVISION = "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"
TRAIN_DATASET = "geo3k-train"
EVAL_DATASET = "geo3k-validation"
SMOKE_TRAIN_DATASET = "geo3k-smoke-train"
SMOKE_EVAL_DATASET = "geo3k-smoke-validation"


class GeometryProblem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    problem: str
    images: list[Image.Image] = Field(min_length=1)
    answer: str


def row_to_task(row: GeometryProblem) -> Task:
    content: list[dict[str, JsonValue]] = [{"type": "text", "text": row.problem.replace("<image>", "").strip()}]
    for image in row.images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    task_id = hashlib.sha256(json.dumps([content, row.answer], sort_keys=True).encode()).hexdigest()[:24]
    return Task(id=task_id, input=[Message(role="user", content=content)], grader_params={"answer": row.answer})
