# Running an eval remotely

Remote is the default for `client.eval.run(config)`. AC2 discovers the named
components, builds the managed project, uploads it, and runs the eval on a
cluster.

## Config module

```python
# src/my_project/eval.py
from ac2.sdk import Client, EvalConfig

CONFIG = EvalConfig(
    agent="MyAgent",
    env="MyEnvironment",
    grader="MyGrader",
    dataset="qa-dataset",
    num_samples=1,
    max_parallel=8,
    cluster_id="ac-jurassic",   # optional; org default if omitted
    name="qa-eval",
)


async def main() -> None:
    await Client(project="my-project").eval.run(CONFIG)
```

Remote evals require a registered dataset — inline `tasks=` is local-only.

## Launch

```bash
ac2 secrets put --key OPENAI_API_KEY --value <val>
uv run python -m my_project.eval
```

SDK equivalent from inside the project folder:

```python
run = await Client(project="my-project").eval.run(CONFIG)
print(run.eval_id)
```

## Prerequisites

1. `ac2 project init` so `[tool.ac2]` exists and the project is registered.
2. Runtime deps declared in `[project].dependencies` (packaged into the wheel).
3. Dataset uploaded (`references/datasets.md`).
4. Provider secrets via `ac2 secrets put` (remote pods do not read local `.env`).

## Monitor

```bash
ac2 evals list
ac2 evals get <eval_id>
ac2 evals logs <eval_id>
```

Dashboard: `https://platform.appliedcompute.com`.
