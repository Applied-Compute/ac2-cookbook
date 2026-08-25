# Submitting a training run

Define a `TrainingConfig` with component class names and pass it to the SDK:

```bash
uv run python -m my_project.train
```

SDK equivalent:

```python
from ac2.sdk import Client
from my_project.train import CONFIG

run = Client(project="my-project").train.run(CONFIG)
print(run.train_id)
```

## Config module

```python
# src/my_project/train.py
from ac2.sdk import Client, TrainingConfig

CONFIG = TrainingConfig(
    model="Qwen/Qwen3-4B",
    n_training_replicas=4,
    n_inference_replicas=4,
    samples_per_problem=4,
    problem_batch_size=8,
    num_train_steps=20,
    ac2_agent="MyAgent",
    ac2_env="MyEnvironment",
    ac2_grader="MyGrader",
    ac2_train_dataset="my-train-dataset",
    ac2_eval_dataset="my-eval-dataset",
    training_agent_names=["MyAgent"],
    cluster_id="ac-jurassic",
    name="my-train",
)


def main() -> None:
    Client(project="my-project").train.run(CONFIG)
```

Optional: `ac2_user=` for a simulated user policy (see `tau2bench`).

## Prerequisites

1. Working local and remote eval with the same components and grader.
2. Managed project with one importable package under `src/`.
3. Train and eval datasets uploaded.
4. Provider secrets registered (`ac2 secrets put`).
5. Cluster access when required by your org.

## Monitor

```bash
ac2 train list
ac2 train get <train_id>
ac2 train logs <train_id>
ac2 train checkpoints <train_id>
```

## Key `TrainingConfig` fields

Required: `model`, `n_training_replicas`, `n_inference_replicas`,
`samples_per_problem`, `problem_batch_size`, either `ac2_orchestrator` or both
`ac2_agent` and `ac2_env`,
`ac2_train_dataset`.

Common optional: `ac2_grader`, `ac2_eval_dataset`, `ac2_user`,
`num_train_steps`, `cluster_id`, `backend`, `name`, `priority`, `tags`.
