# Datasets

A **dataset** is a named, project-scoped collection of `Task` objects stored on AC2. Once uploaded, you reference the dataset by name from any eval or training run. Datasets are deduped and versioned — re-uploading the same task is a no-op, and you can read historical membership with `as_of=`.

This is the single place that owns dataset semantics. `eval-remote.md` and `training.md` both require a dataset (inline tasks aren't supported there); `eval-local.md` accepts inline tasks but datasets are the right choice once the task set stabilizes.

## Define tasks

A `Task` is one rollout target: initial messages, optional env params, optional grader params.

```python
from ac2.runtime import Message, Task

tasks = [
    Task(
        input=[Message(role="user", content="What is the capital of France?")],
        env_params={},                                  # optional; passed to env.setup
        grader_params={"expected_answer": "Paris"},     # optional; passed to grader
    ),
    Task(
        input=[Message(role="user", content="What is 15 + 27?")],
        grader_params={"expected_answer": "42"},
    ),
]
```

Tasks can also carry an explicit `id=` for stable identifiers across reruns; if you omit it, the platform derives one from content. For larger task sets it's worth setting `id` from a hash of the input + grader params so that you can join runs back to specific problems.

```python
import hashlib

def make_task_id(question: str, answer: str) -> str:
    return hashlib.sha256(f"{question}|{answer}".encode()).hexdigest()[:16]

task = Task(
    id=make_task_id(question, answer),
    input=[Message(role="user", content=question)],
    env_params={"answer": answer},
    grader_params={"answer": answer},
)
```

## Create a dataset and add tasks

`add_tasks` uploads tasks and links them to the dataset. It's safe to run repeatedly — duplicate content is deduped.

```python
from ac2.sdk import Client

client = Client(project="my-project")
client.projects.get()

client.datasets.create("qa-dataset")
client.datasets.add_tasks(dataset="qa-dataset", tasks=tasks)
```

### Get-or-create pattern

The most common script shape — works whether the dataset already exists or not:

```python
from ac2.sdk import Client, DatasetNotFoundError

client.projects.get()

try:
    client.datasets.get("qa-dataset")
except DatasetNotFoundError:
    client.datasets.create("qa-dataset")

client.datasets.add_tasks("qa-dataset", tasks=tasks)
```

### Loading tasks from a file

When tasks come from a parquet/CSV/JSON file, the standard pattern is one function that maps a row to a `Task`, then list-comprehend over the rows. Example for a parquet with `prompt` + `reward_model.ground_truth` columns:

```python
import pandas as pd

def row_to_task(row: pd.Series) -> Task:
    question = row["prompt"]
    answer = str(row["reward_model"]["ground_truth"])
    return Task(
        id=make_task_id(question, answer),
        input=[Message(role="user", content=question)],
        env_params={"answer": answer},
        grader_params={"answer": answer},
    )

df = pd.read_parquet("path/to/file.parquet")
tasks = [row_to_task(row) for _, row in df.iterrows()]
client.datasets.add_tasks("qa-dataset", tasks=tasks)
```

## List and inspect

```python
client.datasets.list()                              # all datasets in the project
client.datasets.get("qa-dataset")                   # single dataset metadata

page = client.datasets.list_tasks("qa-dataset", limit=100)
for ref in page:
    print(ref.id)
```

To page through tasks with their decoded `input` / `env_params` / `grader_params` inlined:

```python
page = client.datasets.list_tasks_with_content("qa-dataset", limit=100)
for task in page:
    print(task.input, task.grader_params)
```

To iterate every task in a dataset (handles paging for you):

```python
for task in client.datasets.load_tasks("qa-dataset"):
    print(task.input)
```

## Remove and delete

Removing tasks updates dataset membership while preserving historical run reproducibility.

```python
client.datasets.remove_tasks(dataset="qa-dataset", tasks=tasks)
client.datasets.delete("qa-dataset")
```

## Versioning

Dataset membership is recorded with timestamps. Pass `as_of=` to read the dataset as it existed at a past time — useful for reproducing an older eval or training run:

```python
from datetime import datetime, timezone

snapshot = datetime(2026, 5, 1, tzinfo=timezone.utc)
page = client.datasets.list_tasks("qa-dataset", as_of=snapshot)
```

`client.datasets.active_tasks(dataset)` returns the currently-active task references used by remote eval and training submissions.

## Use a dataset in a run

Put the dataset name (or `DatasetSource`) in the project's eval or training config.

### Eval

```python
# src/my_project/eval.py
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

```bash
uv run python -m my_project.eval --local   # inline tasks= also allowed locally
uv run python -m my_project.eval           # dataset required remotely
```

### Training — train + eval datasets

```python
# src/my_project/train.py
CONFIG = TrainingConfig(
    # ... model + replica counts ...
    ac2_agent="MyAgent",
    ac2_env="MyEnvironment",
    ac2_grader="MyGrader",
    ac2_train_dataset="my-train-dataset",
    ac2_eval_dataset="my-eval-dataset",
    training_agent_names=["MyAgent"],
)
```

Typical pattern: keep `upload_dataset.py` at the project root so a fresh checkout can rebuild datasets deterministically.

## Things to keep in mind

- **Datasets are project-scoped.** A dataset named `qa-dataset` in `project-a` is not the same as `qa-dataset` in `project-b`.
- **Tasks are deduped by content.** Re-running `add_tasks` with the same tasks is a no-op. Want to "update" a task? Build a new one and remove the old one from the dataset; past runs remain reproducible.
- **`id=` is optional but useful.** Setting a stable `id` (e.g. from a hash) lets you join eval results back to specific problems across reruns.
- **Membership is versioned with timestamps.** `as_of=` reads historical state. `client.datasets.active_tasks(name)` returns the current snapshot used by remote eval and training.
- **`add_tasks` parallelizes uploads.** For large task sets (10k+) this is fast, but it still does real network work. Don't loop `add_tasks` once per task — batch.
- **Inline tasks are only for local eval.** Remote eval and training need a registered dataset name.
- **Project not found** → run `ac2 project init` inside the project folder, then rerun.
- **`DatasetNotFoundError` on `datasets.get(...)`** → the dataset doesn't exist in this project. Use the get-or-create pattern above.
- **Tasks "disappear" after a re-upload** → you `delete`d the dataset (which clears membership) or you're reading at an `as_of=` timestamp before your most recent `add_tasks`.
- **Remote eval/training run sees 0 tasks** → the dataset is empty for the snapshot at submission time. Check `client.datasets.active_tasks(name)` locally.
- **Tasks with the same input but different expected answers collide** → they don't, because deduping includes `grader_params` and `env_params`. But if you're using a stable `id=` derived from input only, two distinct tasks will end up sharing an ID. Hash the full set of fields you care about.
