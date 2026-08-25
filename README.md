# AC2 Cookbook

Starter projects for building, evaluating, deploying, and training agents on
[AC2](https://docs.appliedcompute.com/).

Each folder is a complete AC2 project. Runtime components are discovered from
its `src/` package and launched by class name from normal Python config modules.
Start with the project that matches the pattern you want to learn.

## Getting Started

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

1. Enter a project, register it, and sync deps:
   ```bash
   cd dapo-math-check
   ac2 project init
   uv sync
   ```
2. Follow that project's README for dataset setup and runnable eval, train, and
   deployment modules.


## Core examples

| Project                                     | Shows                                                                                  | Use It For                       |
| ------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------- |
| [rock-paper-scissors](rock-paper-scissors/) | Local sessions, tool use, grading, deployment                                          | A small end-to-end AC2 agent     |
| [dapo-math-check](dapo-math-check/)         | Datasets, custom user policy, stateful tools, eval, training                           | A focused train/eval starter     |
| [tau2bench](tau2bench/)                     | Simulated customer user, domain environments, multi-part grader, remote eval, training | A larger benchmark-style starter |

## Bring your own harness examples

Bring your own harness (BYOH) connects an existing agent runtime to AC2 for
evals and training. A harness accepts a task, prepares its environment, runs an
agent and its tools, and returns the result. One attempt at one task is a
rollout.

The examples cover both BYOH deployment models:

- **AC2-managed lifecycle:** you write a Python orchestrator that AC2 runs
  inside the job. Start here when your harness is a CLI or library.
- **Self-hosted lifecycle:** you run a three-endpoint Harness API in your own
  network. An unmodified AC2 Harness Connector forwards lifecycle requests to
  it over private HTTP.

These examples use one user turn per rollout and run grading in AC2. For a
self-hosted rollout, the initial `/submit` request does not contain a rollout
ID. Your Harness API creates the ID, returns it to AC2, and accepts the same ID
in `/get_status` and `/collect_outputs`.

Read the [BYOH overview](https://docs.appliedcompute.com/platform/custom-harnesses)
for the vocabulary, architecture, and selection guide before running an
example.

BYOH is currently in beta. Contact the AC team to enable it for your AC2
project before running these examples. We will provision the required AC2
infrastructure, including model relay access.

| Project                                                               | Lifecycle   | Shows                                                        |
| --------------------------------------------------------------------- | ----------- | ------------------------------------------------------------ |
| [swebench-opencode](swebench-opencode/)                               | AC2-managed | A CLI harness, sandbox handoff, and AC2-side grading          |
| [byoh-agents-web-search-api](byoh-agents-web-search-api/)             | Self-hosted | A stateful Harness API with AC2-side grading                  |
| [byoh-modal-coding-api](byoh-modal-coding-api/)                       | Self-hosted | External execution with customer-side grading                |


## Helpful Links

- [AC2 docs](https://docs.appliedcompute.com/)
- [AC2 platform](https://platform.appliedcompute.com/)
