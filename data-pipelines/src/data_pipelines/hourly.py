from datetime import UTC, datetime, timedelta

from ac2.runtime import Task
from ac2.sdk.pipelines import PipelineSpec, pipeline, step

from data_pipelines.monitor import ScoredConversation, grade_traces
from data_pipelines.traces import read_traces


@step
def select_failures(*, results: list[ScoredConversation]) -> list[Task]:
    return [
        Task(
            input=list(result.conversation.prompt),
            grader_params={"required_format": "answer-confidence-json"},
            tags={"source_trace_id": result.conversation.trace_id, "selection_reason": result.reasoning},
        )
        for result in results
        if result.score < 1
    ]


@pipeline(
    name="hourly-trace-failures",
    log_prints=True,
    ac2=PipelineSpec(
        secret_keys=("TRACE_READER_API_KEY",),
        output_dataset_name="hourly-trace-failures",
        schedule_cron_expression="0 * * * *",
        max_concurrent_runs=1,
        max_runtime_seconds=900,
    ),
)
def hourly_trace_failures(*, project: str = "<project>", source_job_id: str = "<source-job-id>") -> list[Task]:
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    conversations = read_traces(project=project, source_job_id=source_job_id, start=end - timedelta(hours=1), end=end)
    results = grade_traces(conversations=conversations)
    tasks = select_failures(results=results)
    print(f"Selected {len(tasks)} tasks from {len(results)} eligible conversations")
    return tasks
