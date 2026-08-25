---
name: ac2-companion
description: Helps users build, evaluate, deploy, and train agents on Applied Compute Agent Cloud (AC2). Use when the user mentions AC2, the `ac2` CLI, `ac2.runtime`/`ac2.sdk`/`ac2.tracing`, AC2 primitives (`Agent`/`Environment`/`Grader`/`Orchestrator`/`Task`), `EvalConfig`/`TrainingConfig`/`DeploymentConfig`, `ac2 eval run`/`ac2 train run`/`ac2 deployments run`, `client.eval.run`/`client.train.run`/`client.deployments.create`, dispatching to AC2 clusters (e.g. `ac-jurassic`), or `platform.appliedcompute.com` — even when "AC2" isn't said explicitly.
argument-hint: "[task or question]"
allowed-tools: Bash(ac2 config*) Bash(ac2 project*) Bash(ac2 eval*) Bash(ac2 train*) Bash(ac2 deployments*) Bash(ac2 jobs list*) Bash(ac2 datasets*) Bash(ac2 secrets*) Bash(uv run python*)
user-invocable: true
---

# AC2 companion

Help the user build, evaluate, deploy, and train agents on AC2 (Applied Compute Agent Cloud) using the `ac2` SDK and CLI.

## How AC2 works

AC2 is a platform for building, evaluating, deploying, and post-training agents. Work is organized around **managed projects**: a Python package under `src/` whose runtime components AC2 discovers and registers by class name. Define ordinary Python modules containing `EvalConfig`, `TrainingConfig`, or `DeploymentConfig`, then pass those configs to the matching `Client` method. Eval can run locally or remotely.

- **Serving** — deploy an agent and open streaming sessions against it. `client.session.from_local`, `ac2 deployments run` / `client.deployments.create`, `client.session.from_remote`.
- **Datasets** — upload, version, and share task sets. `client.datasets.*` / `upload_dataset.py`.
- **Evaluation** — score an agent against a dataset with a grader. `ac2 eval run` / `client.eval.run(local=...)`.
- **Training** — post-train an agent on a dataset against a grader. `ac2 train run` / `client.train.run`.

The same runtime primitives (`Agent`, `Environment`, `Grader`, optionally `OrchestratorProtocol`) compose into every one of these — don't fork them per workflow.

The documentation index lives at <https://docs.appliedcompute.com/llms.txt> — fetch it to discover all available pages before exploring further.

## Operating rules

Prefer the AC2-native path before custom scaffolding:

1. Define the task contract: prompt shape, tool I/O, stop conditions, grading output, and success metrics.
2. Use the smallest correct AC2 primitive: `Agent` for model behavior, `Environment` for tools and per-rollout state, `Task` for inputs/params, `Grader` for scoring, and `OrchestratorProtocol` only for multi-agent or custom control flow.
3. Smoke-test locally before remote work: local session for deployable agents and a local eval for eval/training candidates.
4. Promote to platform workflows only after the local loop is stable: upload datasets, register secrets, run remote eval, then train.
5. Record exact commands, project, dataset, model, cluster, and any unresolved assumptions in the handoff.

Do not hide platform or framework mismatches. If a construct from another framework does not map cleanly to AC2, explain the mismatch and use [migrations.md](references/migrations.md) before writing compatibility glue.

## Quick start

`ac2.runtime` is the Python library for agents, environments, orchestrators, and graders. The shape of every AC2 program is the same:

1. Create an `Agent` (or subclass for custom hooks).
2. Create an `Environment` (or subclass with `@tool` methods).
3. Reference their exact class names in `EvalConfig`, `TrainingConfig`, or `DeploymentConfig`.
4. Pass the config explicitly to the matching `Client` method from the managed project folder.

A complete local interactive loop:

```python
import asyncio

from ac2.runtime import Agent, Environment, ModelConfiguration
from ac2.sdk.client import Client


class ExampleAgent(Agent):
    description = "Answers questions."
    model_configuration = ModelConfiguration(model="gpt-4o-mini")
    system_prompt = "Answer concisely."


class ExampleEnvironment(Environment):
    pass

client = Client(project="my-project")
session = client.session.from_local(agent=ExampleAgent(), env=ExampleEnvironment())


async def main() -> None:
    async with session:
        while True:
            text = input("\n> ")
            async for event in session.run(text):
                if event.type == "text_delta":
                    print(event.content, end="", flush=True)
            print()


asyncio.run(main())
```

Managed projects use the standard package layout from `ac2 project init`:

```text
my-project/
  pyproject.toml          # project metadata + [tool.ac2]
  upload_dataset.py
  src/my_project/
    agent.py
    environment.py
    grader.py
    eval.py               # CONFIG + runnable main()
    train.py              # CONFIG + runnable main()
    deploy.py             # CONFIG + runnable main()
```

## Where to find more

Read only the reference file(s) the user's task actually needs.

- **Setup & install**: see [setup.md](references/setup.md) — install `ac2`, configure the CLI, `ac2 project init`
- **Defining primitives**: see [runtime.md](references/runtime.md) — `Agent`, `Environment`, `Grader`, `Orchestrator`, `Task`
- **Migrations**: see [migrations.md](references/migrations.md) — translate Claude Agents SDK, OpenAI Agents SDK, Prime Verifiers, Silverback, and generic harnesses to AC2
- **Datasets**: see [datasets.md](references/datasets.md) — upload, version, list (`client.datasets.*`)
- **Local eval**: see [eval-local.md](references/eval-local.md) — `EvalConfig` + `ac2 eval run --local`
- **Remote eval**: see [eval-remote.md](references/eval-remote.md) — `EvalConfig` + `ac2 eval run`
- **Training**: see [training.md](references/training.md) — `TrainingConfig` + `ac2 train run`
- **Deploy**: see [deploy.md](references/deploy.md) — `DeploymentConfig` + `ac2 deployments run`
- **CLI**: see [cli.md](references/cli.md) — `ac2` command reference
- **Troubleshooting**: see [troubleshooting.md](references/troubleshooting.md) — symptom → cause map; check this when something fails

Datasets show up across eval-remote, training, and (optionally) eval-local — load [datasets.md](references/datasets.md) alongside those workflows when the user needs to upload or share task sets.

## Workflow order

For a new AC2 user, follow [setup.md](references/setup.md) → [runtime.md](references/runtime.md) → [eval-local.md](references/eval-local.md) → [datasets.md](references/datasets.md) → [eval-remote.md](references/eval-remote.md) → [training.md](references/training.md). [deploy.md](references/deploy.md) is orthogonal — reach for it whenever the user wants an interactive endpoint, not just at the end.

For a migration, start with [migrations.md](references/migrations.md), then read the runtime/eval/deploy/training reference that matches the target workflow. Preserve the old task contract first; only then simplify into idiomatic AC2.

For examples in this cookbook, use the existing runnable projects:

- `rock-paper-scissors` — local/remote session and deployment with tools.
- `dapo-math-check` — custom user policy, stateful bounded tool, datasets, and training.
- `tau2bench` — simulated customer user, domain environments, datasets, remote eval, and training.
