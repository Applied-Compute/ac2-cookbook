from datetime import UTC, datetime, timedelta

from ac2.sdk.pipelines import PipelineSpec, pipeline, step
from pydantic import BaseModel, Field, ValidationError

from data_pipelines.traces import Conversation, read_traces


class Answer(BaseModel):
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ScoredConversation(BaseModel):
    conversation: Conversation
    score: float
    reasoning: str


@step
def grade_traces(*, conversations: list[Conversation]) -> list[ScoredConversation]:
    results: list[ScoredConversation] = []
    for conversation in conversations:
        try:
            Answer.model_validate_json(conversation.answer)
        except ValidationError:
            score, reasoning = 0.0, "Answer does not match the required JSON schema"
        else:
            score, reasoning = 1.0, "Answer matches the required JSON schema"
        results.append(ScoredConversation(conversation=conversation, score=score, reasoning=reasoning))
    return results


@pipeline(
    name="monitor-production",
    log_prints=True,
    ac2=PipelineSpec(secret_keys=("TRACE_READER_API_KEY",), max_runtime_seconds=900),
)
def monitor_production(*, project: str, source_job_id: str) -> None:
    end = datetime.now(UTC)
    conversations = read_traces(project=project, source_job_id=source_job_id, start=end - timedelta(hours=1), end=end)
    results = grade_traces(conversations=conversations)
    failures = sum(result.score < 1 for result in results)
    print(f"JSON format violations: {failures}/{len(results)} eligible conversations")
