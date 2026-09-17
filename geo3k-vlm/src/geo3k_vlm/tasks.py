import base64
import hashlib
import io
import json

from ac2.runtime import Message, Task

SOURCE = "hiyouga/geometry3k"
SOURCE_REVISION = "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7"
TRAIN_DATASET = "geo3k-train"
EVAL_DATASET = "geo3k-validation"


def row_to_task(row: dict) -> Task:
    content = [{"type": "text", "text": row["problem"].replace("<image>", "").strip()}]
    for image in row["images"]:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    if len(content) == 1:
        raise ValueError("Geometry3K tasks require an image")
    answer = str(row["answer"])
    task_id = hashlib.sha256(json.dumps([content, answer], sort_keys=True).encode()).hexdigest()[:24]
    return Task(id=task_id, input=[Message(role="user", content=content)], grader_params={"answer": answer})
