# `ac2` CLI reference

## `ac2` auth / config

| Command | Description |
|---------|-------------|
| `ac2 login` | Interactive credential setup |
| `ac2 logout` | Clear credentials |
| `ac2 config show` / `set` / `unset` | Inspect or change local config |
| `ac2 upgrade` | Upgrade the CLI/SDK from the AC2 package index |

## `ac2 project`

| Command | Description |
|---------|-------------|
| `ac2 project init [name]` | Create `./<name>/` or adopt the current folder; register + write `[tool.ac2]` |
| `ac2 project push` | Snapshot and upload the managed project |
| `ac2 project pull <project> [--version <hash>]` | Download source into `./<project>/` |
| `ac2 --project <project> project versions` | List recorded code versions |

## Run project workflows

Run these from inside a folder with `[tool.ac2]`:

| Command | Description |
|---------|-------------|
| `uv run python -m my_project.eval` | Run the project's eval config remotely |
| `uv run python -m my_project.eval --local` | Run the same eval locally |
| `uv run python -m my_project.train` | Submit the project's training config |
| `uv run python -m my_project.deploy` | Submit the project's deployment config |

Launch options belong in `EvalConfig` / `TrainingConfig` / `DeploymentConfig`,
not as CLI flags (except `--local` for eval).

## Listing / inspecting

List commands show resources across projects you can access. Pass `-p` to filter:

| Command | Description |
|---------|-------------|
| `ac2 evals list` / `get` / `logs` | Eval runs |
| `ac2 train list` / `get` / `logs` / `checkpoints` / `deploy` | Training runs |
| `ac2 deployments list` / `get` / `logs` / `stop` / `delete` | Deployments |
| `ac2 jobs list` / `stop` / `delete` | Dispatch jobs |
| `ac2 secrets put` / `list` / `delete` | User secrets |
| `ac2 datasets …` | Dataset management |

## ID prefixes

| Prefix | Resource |
|--------|----------|
| `dep_…` | Deployment |
| `job_…` | Remote job |
| `eval_…` | Eval run |
| `train_…` | Training run |

## CLI vs SDK

- **CLI**: init/push/pull and list/inspect/cleanup.
- **SDK**: `Client(project=...)` for datasets, sessions, and programmatic launches.
  Pass `EvalConfig`, `TrainingConfig`, or `DeploymentConfig` explicitly to
  `client.eval.run(config)`, `client.train.run(config)`, or
  `client.deployments.create(config)`.
