# Data pipelines

These projects show the three data-pipeline patterns in the [AC2 pipelines guide](https://docs.appliedcompute.com/platform/pipelines): production monitoring, hourly trace grading, and synthetic tasks from public GitHub pull requests.

## Setup

Install the AC2 CLI, authenticate, then run from this directory:

```bash
uv sync
ac2 project init
```

Set your source trace job or inference deployment ID and AC2 project in `src/data_pipelines/hourly.py` before deploying the scheduled example. Choose a source in a project your API key can read. Register secrets once:

```bash
uv run python -c 'from getpass import getpass; from ac2.sdk import Client; Client(project="<project>").secret.put(values={"TRACE_READER_API_KEY": getpass("AC2 trace-reader API key: ")})'
uv run python -c 'from getpass import getpass; from ac2.sdk import Client; Client(project="<project>").secret.put(values={"OPENAI_API_KEY": getpass("OpenAI API key: ")})'
```

Each pipeline declares the secret it needs. Values are injected into the remote run and are not pipeline parameters. The trace key must belong to a user who can export the selected source project's traces.

## Push, deploy, and run

Push the project to register its pipeline definitions and create immutable revisions:

```bash
ac2 project push
```

Deploy a named revision and run a pipeline with the SDK:

```python
from ac2.sdk import Client

client = Client(project="<project>")
revisions = client.pipelines.list_revisions("monitor-production", order="newest")
revision = revisions.items[0]
client.pipelines.deploy("monitor-production", revision_id=revision.id)
run = client.pipelines.run(
    "monitor-production",
    parameters={"project": "<source-project>", "source_job_id": "<source-job-id>"},
)
print(run.id)
```

For `hourly-trace-failures`, the deployed cron runs hourly at the start of the hour (UTC); edit the source defaults to point to your trace source before pushing. A completed run replaces that pipeline's named output dataset with its results, including an empty result if no tasks pass the filter. Review the dataset before using it for training.

To disable the schedule, set `schedule_cron_expression=None` in `hourly.py`, push, and deploy the new revision. Cancelling a run does not stop its schedule.

## Projects

- `monitor.py` checks completed production responses for a required JSON shape and logs the violation count. It does not write a dataset.
- `hourly.py` exports the previous completed UTC hour, grades format compliance, keeps failed prompts, and replaces the `hourly-trace-failures` dataset each run. This rolling snapshot does not backfill late traces.
- `github.py` samples up to three recently updated merged pull requests from a public repository, generates a review task and rubric, tests a baseline LLM answer, and saves tasks that score below 0.8. “Failed” means the model failed the generated rubric; the example does not check out or execute repository code. Review generated tasks before training.

The GitHub example sends public PR text and patches to OpenAI using your registered key and incurs API usage. Keep the sample bounded, and only use repositories and keys you are authorized to use.
