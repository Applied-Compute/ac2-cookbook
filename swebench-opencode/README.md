# OpenCode on SWE-bench with AC2-managed lifecycle

This project runs an unmodified OpenCode harness against AC2. OpenCode owns the
agent loop and tools inside a Modal sandbox. AC2 provides an OpenAI-compatible
model relay. OpenCode sends model requests to this endpoint; AC2 routes them to
the model selected for the job and records the trace. AC2 grades the resulting
repository with the official SWE-bench harness. The project-defined
orchestrator runs inside the AC2 job and owns the rollout lifecycle.

The example works for both evals against an API model and reinforcement
learning with Qwen3-4B.

## What this example teaches

This is the
[AC2-managed lifecycle](https://docs.appliedcompute.com/platform/custom-harnesses/ac2-managed)
for a **stateful, sandbox-heavy CLI harness**:

- Your orchestrator runs inside the AC2 job.
- It starts OpenCode as a CLI in one Modal sandbox and tracks the process in
  memory.
- After the rollout, it copies the code patch into a clean grading sandbox.
- The official SWE-bench grader runs in AC2 against that reconstructed state.

There is no Harness API server or Harness Connector. This is the right pattern
when AC2 can run the Python adapter and the underlying harness is an executable
or library.

## What you implement

[`src/swebench_opencode/orchestrator.py`](src/swebench_opencode/orchestrator.py)
is the code you implement. Its
`SwebenchOpenCodeOrchestrator` maps the three lifecycle operations to OpenCode:

- `trigger_rollout` starts the CLI with the relay URL and API key supplied by
  AC2. The orchestrator generates and returns the rollout ID; AC2 does not
  assign it.
- `check_status` translates the CLI process state into `RUNNING`, `COMPLETED`,
  or `ERRORED`.
- `fetch_assets` captures the model's patch and prepares the environment for
  grading.

`environment.py` owns the Modal sandboxes because they are part of this
harness's execution and grading state.

## How a rollout moves through the example

1. AC2 creates `SwebenchOpenCodeEnvironment` for a dataset task.
2. The environment starts a Modal sandbox from the task's SWE-bench image.
3. AC2 calls `trigger_rollout`; OpenCode works in the repository and sends model
   requests through the relay.
4. AC2 polls `check_status` until the CLI exits.
5. `fetch_assets` extracts the patch and applies it to a clean grading sandbox.
6. `SwebenchVerifiedGrader` runs the official tests and returns a binary score.

## Code map

| File | Responsibility |
| --- | --- |
| `src/swebench_opencode/orchestrator.py` | Your adapter around the OpenCode CLI |
| `src/swebench_opencode/environment.py` | Harness and grading sandboxes plus the patch handoff |
| `src/swebench_opencode/grader.py` | Official SWE-bench grading inside AC2 |
| `upload_dataset.py` | Public task images and AC2 Blob preparation |
| `src/swebench_opencode/eval.py` | Eval configuration |
| `src/swebench_opencode/train.py` | Bounded GRPO training configuration |

## Adapt it to your harness

Replace the OpenCode command and task-specific sandbox setup with your CLI or
library. Preserve the three orchestrator methods, pass the relay credentials to
your model client unchanged, and store everything the grader needs on the
orchestrator environment during `fetch_assets`.

## Setup

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

From this directory:

```bash
ac2 project init
uv sync
uv run modal setup
```

Add your Modal credentials to the project:

```bash
ac2 secrets put --key MODAL_TOKEN_ID --value <token-id>
ac2 secrets put --key MODAL_TOKEN_SECRET --value <token-secret>
```

BYOH is currently in beta. Contact the AC team to enable it for this project.
We will provision the required model relay and project configuration.

For evals against an OpenAI model, store the provider key in the project:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

## Upload tasks

The uploader reads the public SWE-bench Verified dataset, uses the public
`swebench/sweb.eval...` task images, and stores each instruction directory as
an AC2 Blob:

```bash
uv run python upload_dataset.py --num-tasks 2
```

Pass `--instance-id <id>` one or more times to select specific tasks.

Commit and push this project before launching a remote job so AC2 can load its
components. The eval and training launchers use the model access provisioned by
AC2 automatically.

## Run an eval

```bash
uv run python -m swebench_opencode.eval
```

## Run a bounded training smoke test

```bash
uv run python -m swebench_opencode.train
```

The default is deliberately small: one Qwen3-4B training step, one problem per
batch, two samples, and a global sampling concurrency of two. Increase those
values only after the smoke test succeeds.
