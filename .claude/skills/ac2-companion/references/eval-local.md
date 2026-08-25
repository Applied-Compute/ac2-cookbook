# Running an eval locally

Define an `EvalConfig` with component class names and pass it to the SDK:

```bash
uv run python -m my_project.eval --local
```

Or from Python inside a managed project folder:

```python
from ac2.sdk import Client
from my_project.eval import CONFIG

run = await Client(project="my-project").eval.run(CONFIG, local=True)
```

The agent loop, environment, and grader execute in the local process. Tracing
and the eval-job record still go to the platform.

## Define the config module

```python
# src/my_project/eval.py
from ac2.sdk import Client, EvalConfig

CONFIG = EvalConfig(
    agent="MyAgent",
    env="MyEnvironment",
    grader="MyGrader",
    dataset="qa-dataset",          # or tasks=[...] for ad-hoc local runs
    num_samples=1,
    max_parallel=4,
)


async def main() -> None:
    await Client(project="my-project").eval.run(CONFIG, local=True)
```

Pass `agent` + `env`, or `orchestrator` for custom control flow. Set `dataset`
or `tasks` (inline `tasks` are local-only).

## Cap dataset size

```python
from ac2.sdk import DatasetSource, EvalConfig

CONFIG = EvalConfig(
    agent="MyAgent",
    env="MyEnvironment",
    grader="MyGrader",
    dataset=DatasetSource(dataset="qa-dataset", num_tasks=10),
    num_samples=1,
    max_parallel=4,
)
```

## Prerequisites

1. Managed project initialized with `ac2 project init` and one package under `src/`.
2. Local provider credentials (`OPENAI_API_KEY` in the shell or `.env`).
3. A dataset uploaded, or inline `tasks=` for a quick smoke test.

## What "local" means

- Workload runs on the user's machine.
- Results and traces register on the platform.
- Cluster / backend / remote-only fields on `EvalConfig` must not be set.

Next: upload a shared dataset (`references/datasets.md`) and run remotely
(`references/eval-remote.md`).
