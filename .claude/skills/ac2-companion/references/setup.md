# Setup: install, configure, scaffold

Two things have to happen, in order: install the `ac2` CLI/SDK (the installer
authenticates), then initialize a project. Agents can follow
<https://platform.appliedcompute.com/agents.md> instead of this file.

## Quick path

```bash
# 1. Install the CLI (also installs uv if needed and logs in via the console).
curl -fsSL https://api.appliedcompute.com/install.sh | sh

# 2. Scaffold a managed project.
ac2 project init my-agent
cd my-agent
uv sync
cp .env.example .env   # set OPENAI_API_KEY=...

# 3. Upload the starter dataset and run the eval module.
uv run python upload_dataset.py
uv run python -m my_agent.eval --local
```

## Project-based install (`pyproject.toml`)

Cookbook projects already pin the AC2 index. After install:

```bash
uv sync
```

If sync fails on auth:

```bash
export UV_INDEX_AC2_USERNAME=__token__
export UV_INDEX_AC2_PASSWORD="$(python3 -c 'import json, os; print(json.load(open(os.path.expanduser("~/.ac2/config")))["api_key"])')"
uv sync
```

## Configure the CLI

The installer completes login. If a later command reports missing credentials,
re-authenticate with `ac2 login`. Inspect local config with:

```bash
ac2 config show
```

Env vars (preferred for CI):

| Variable | Description |
|----------|-------------|
| `AC_API_KEY` / `AC2_API_KEY` | API key |
| `AC_BASE_URL` / `AC2_BASE_URL` | Platform URL |
| `AC2_CONFIG_PATH` | Override config file path |

## Initialize a project

```bash
ac2 project init my-agent   # creates ./my-agent/
# or, inside an existing folder:
ac2 project init
```

This registers the project, adds `[tool.ac2]` to `pyproject.toml`, and creates
the package scaffold. AC2 discovers component subclasses recursively from the
single importable package under `src/`. Layout:

```text
my-project/
  .ac2ignore
  pyproject.toml
  upload_dataset.py
  src/
    my_project/
      agent.py
      environment.py
      grader.py
      eval.py      # EvalConfig + runnable main()
      train.py     # TrainingConfig + runnable main()
      deploy.py    # DeploymentConfig + runnable main()
```

Project metadata in `pyproject.toml`:

```toml
[project]
name = "my-agent"
version = "0.1.0"
requires-python = ">=3.12"

[tool.ac2]
```

Declare runtime dependencies in `[project].dependencies`. AC2 builds a wheel
from the package before uploading it for remote eval, train, and deploy.

## Launch project modules

From inside the project folder:

```bash
uv run python -m my_agent.eval              # remote
uv run python -m my_agent.eval --local
uv run python -m my_agent.train
uv run python -m my_agent.deploy
ac2 project push          # snapshot/upload without launching
```

## Secrets

Remote workloads see user-scoped secrets as env vars:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <val>
ac2 secrets list
```

Next: define primitives in `references/runtime.md`.
