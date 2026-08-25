# rock-paper-scissors

Rock-paper-scissors on AC2: the agent plays 40 rounds against a deterministic
opponent, calls `play_move` for each round, and can use `run_python` to inspect
the history.

## Setup

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync
cp .env.example .env   # set OPENAI_API_KEY
```

`ac2 project init` registers this folder and writes the `[tool.ac2]` marker used
by the CLI. Runtime deps live in `[project].dependencies`; AC2 builds a wheel
from this package for remote runs.

## Interactive local session

```bash
OPENAI_API_KEY=<val> uv run python session.py --start
```

## Eval

```bash
uv run python upload_dataset.py
uv run python -m rock_paper_scissors.eval --local
uv run python -m rock_paper_scissors.eval
```

## Deploy

Register your provider key as a secret (remote pods do not read your local
shell), then deploy:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <val>
uv run python -m rock_paper_scissors.deploy
```

Connect to a deployment:

```bash
uv run python session.py --deployment-id <deployment_id>
```

## Layout

```text
rock-paper-scissors/
├── pyproject.toml              # package metadata + [tool.ac2]
├── upload_dataset.py
├── session.py                  # local / remote interactive session
└── src/rock_paper_scissors/
    ├── agent.py
    ├── environment.py
    ├── grader.py
    ├── eval.py                 # named EvalConfig + local/remote launch
    └── deploy.py               # named DeploymentConfig + launch
```
