import io
import os
import time
from datetime import datetime
from tempfile import TemporaryFile

import httpx
import zstandard
from ac2.runtime import Item, Message
from ac2.sdk import Client
from ac2.sdk.pipelines import step
from pydantic import BaseModel, Json, TypeAdapter, ValidationError


class TraceExportError(RuntimeError):
    pass


class SamplingParams(BaseModel):
    n: int | None = None


class RequestAttributes(BaseModel):
    payload_truncated: bool = False
    previous_response_id: str = ""
    response_conversation_id: str = ""
    tools_available: str = ""
    sampling_params: Json[SamplingParams] = SamplingParams()


class Request(BaseModel):
    status: str
    input: str
    output: str
    attributes: Json[RequestAttributes]


class ExportRow(BaseModel):
    trace_id: str
    spans: list[Request]


class Conversation(BaseModel):
    trace_id: str
    prompt: list[Message]
    answer: str


ITEMS = TypeAdapter(list[Item])


def read_conversation(*, line: str) -> Conversation | None:
    try:
        row = ExportRow.model_validate_json(line)
    except ValidationError:
        return None
    if len(row.spans) != 1:
        return None
    request = row.spans[0]
    attributes = request.attributes
    if (
        request.status != "completed"
        or not request.input
        or not request.output
        or attributes.payload_truncated
        or attributes.previous_response_id
        or attributes.response_conversation_id
        or attributes.tools_available not in ("", "[]")
        or (attributes.sampling_params.n or 1) > 1
    ):
        return None
    prompt = ITEMS.validate_json(request.input)
    output = ITEMS.validate_json(request.output)
    if not prompt or not all(isinstance(item, Message) and isinstance(item.content, str) for item in prompt):
        return None
    if len(output) != 1 or not isinstance(output[0], Message):
        return None
    answer = output[0]
    if answer.role != "assistant" or not isinstance(answer.content, str):
        return None
    return Conversation(
        trace_id=row.trace_id,
        prompt=[item for item in prompt if isinstance(item, Message)],
        answer=answer.content,
    )


@step
def read_traces(*, project: str, source_job_id: str, start: datetime, end: datetime) -> list[Conversation]:
    client = Client(project=project, api_key=os.environ["TRACE_READER_API_KEY"])
    state = client.traces.start_export(source_job_id, started_after=start, started_before=end, mode="conversation")
    deadline = time.monotonic() + 300
    while state.status in ("queued", "running"):
        if time.monotonic() >= deadline:
            raise TraceExportError("Trace export did not finish within five minutes")
        if state.artifact_key is None:
            raise TraceExportError("Trace export did not return an artifact key")
        time.sleep(2)
        state = client.traces.export(source_job_id, state.artifact_key)
    if state.status != "ready" or state.download_url is None:
        raise TraceExportError(state.error or "Trace export did not return a download URL")
    conversations: list[Conversation] = []
    with TemporaryFile() as compressed:
        with httpx.stream("GET", state.download_url, timeout=60) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                compressed.write(chunk)
        compressed.seek(0)
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as lines:
                for line in lines:
                    if line.strip():
                        conversation = read_conversation(line=line)
                        if conversation is not None:
                            conversations.append(conversation)
    return conversations

