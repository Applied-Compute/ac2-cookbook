# tau2bench

The tau2-bench customer-service benchmark on AC2. The agent handles airline,
retail, or telecom support tasks while a simulated customer talks to it and a
grader scores the conversation.

## Setup

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync
```

`ac2 project init` registers this folder and writes the `[tool.ac2]` marker used
by the CLI. Runtime deps live in `[project].dependencies`; AC2 builds a wheel
from this package for remote runs.

## Secrets

```bash
ac2 secrets put --key OPENAI_API_KEY --value <val>
ac2 secrets list
```

## Register datasets

```bash
uv run python upload_dataset.py --domain airline
```

Use `--domain retail` or `--domain telecom` for the other domains. This creates
`tau2bench-<domain>-train`, `tau2bench-<domain>-test`, and
`tau2bench-<domain>-base`.

## Run an eval

The eval and train configs default to the airline domain. To switch domains, use
the matching class names and dataset in those modules:

| Domain | Agent | Environment |
|---|---|---|
| airline | `Tau2AirlineAgent` | `AirlineEnvironment` |
| retail | `Tau2RetailAgent` | `RetailEnvironment` |
| telecom | `Tau2TelecomAgent` | `TelecomEnvironment` |

```bash
uv run python -m tau2bench.eval
```

## Submit a training run

```bash
uv run python -m tau2bench.train
```

## Layout

```text
tau2bench/
├── pyproject.toml
├── upload_dataset.py
└── src/tau2bench/
    ├── agent.py                # named agents for every domain
    ├── eval.py                 # named EvalConfig + launch
    ├── train.py                # named TrainingConfig + launch
    ├── user.py
    ├── graders/
    ├── dataloader/
    └── environments/
        ├── airline/
        ├── retail/
        └── telecom/
```

## Notes

- Supported domains: `airline`, `retail`, and `telecom`.
- Upstream policies, tasks, and DB files are fetched lazily from
  [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
  into `src/tau2bench/dataloader/cache/`.
