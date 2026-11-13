# OpenCode on SWE-bench through a self-hosted harness

This example shows the
[self-hosted BYOH deployment model](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted)
for a sandbox-heavy coding harness that also grades its own rollouts.

The Harness API runs an unmodified OpenCode CLI on
[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
tasks in Modal sandboxes and scores each result with the official SWE-bench
tests. OpenCode, the sandboxes, and the grading all run in your infrastructure.
AC2 connects OpenCode to the model selected for the job and records the trace
and the score.

The example has been run as an eval against an API model and as a one-step
Qwen3-4B training smoke test. The smoke test exercises the training path; it
is too small to show learning.

## What this example teaches

This is an **execution and grading in your infrastructure** pattern:

- The Harness API starts OpenCode in one Modal sandbox per rollout and reports
  the CLI process state.
- When OpenCode exits, it diffs the repository against the task image's commit
  and terminates the sandbox. Grading applies that patch in a clean sandbox.
- The official SWE-bench tests run there, and the API returns the score to AC2.
- The AC2 project contains no orchestrator, environment, or grader.

The example keeps rollouts and gradings in memory for readability. In
production, store them in a database so the API can restart or run multiple
replicas.

## What you implement

[`harness_server.py`](harness_server.py) is the server you implement and
operate. Its FastAPI routes are the complete AC2 integration contract:

- `/submit` creates a rollout ID, starts OpenCode in a sandbox with the
  rollout-scoped `ac_api_url` and `ac_api_key`, and returns the ID. It uses
  `Idempotency-Key` to return the same ID if AC2 retries the request.
- `/get_status` translates the CLI process state into `running`, `completed`,
  or `failed` with an error message. A rollout that left no patch because
  OpenCode reported an error or exited non-zero is `failed`, not a score of 0.
  A rollout that exits non-zero but leaves a patch is reported `failed` and is
  still graded on that patch.
- `/collect_outputs` returns the sandbox ID, the patch size, the exit code, and
  the tail of OpenCode's output, or `null` for a rollout that raised.
- `/submit_grading` starts grading the rollout. A patch that fails
  `git apply --check` scores 0 without running the tests.
- `/get_grading_status` reports `running`, `completed`, or `failed`. Grading
  fails when the rollout raised or the test output cannot be parsed.
- `/collect_score` returns the score, the reasoning, and the test results.

The AC2 Harness Connector does not contain your harness logic. It is an
unmodified bridge that opens an outbound WebSocket to AC2 and forwards
requests to these private HTTP endpoints.

## How a rollout moves through the example

1. `eval.py` or `train.py` starts an AC2 job for the dataset.
2. The AC2 job sends `/submit` through the Harness Gateway and Connector.
3. `harness_server.py` starts a Modal sandbox from the task's SWE-bench image,
   writes the issue to `/task/instruction.md`, and runs OpenCode against the
   model endpoint supplied by AC2.
4. When the CLI exits, the server saves the patch and terminates the sandbox.
   The Connector polls `/get_status` until then, and AC2 requests
   `/collect_outputs`.
5. AC2 sends `/submit_grading`. The server verifies a clean grading sandbox,
   applies the patch, and runs the official tests.
6. The Connector polls `/get_grading_status`, then AC2 requests
   `/collect_score` and records the score on the rollout.

The server terminates each sandbox as soon as its rollout or grading ends, so
a rollout that is never graded leaves nothing running. OpenCode is stopped
after 90 minutes.

## Code map

| File | Responsibility |
| --- | --- |
| `harness_server.py` | Your Harness API: OpenCode sandboxes, patch handoff, and SWE-bench grading |
| `upload_dataset.py` | Creates the AC2 dataset from public SWE-bench Verified tasks |
| `src/swebench_opencode/eval.py` | Eval configuration that selects the self-hosted lifecycle |
| `src/swebench_opencode/train.py` | One-step connector-backed GRPO training configuration |

## Adapt it to your harness

Replace the OpenCode command and the sandbox setup in `run_rollout` with your
CLI or scheduler, and replace `grade` with your scoring. Pass `ac_api_url` and
`ac_api_key` to your model client unchanged. The six routes stay the same.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

From this directory, register the project and install both AC2-side and local
Harness API dependencies:

```bash
ac2 project init
uv sync --extra harness
uv run modal setup
```

Modal credentials remain on your machine. For evals against an OpenAI model,
store the provider key in the project:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

## Upload tasks

The uploader reads the public SWE-bench Verified dataset and stores each task
with its datapoint. The server derives the public `swebench/sweb.eval...`
image and the tests from that datapoint.

```bash
uv run python upload_dataset.py --num-tasks 2
```

Pass `--instance-id <id>` one or more times to select specific tasks.

## Start your services

You need two long-running processes.

### 1. Start the Harness API

In the first terminal:

```bash
uv run --extra harness python harness_server.py
```

The API listens on `http://127.0.0.1:8000` by default.

### 2. Start the Harness Connector

The connector uses machine credentials that are separate from your credentials
created by `ac2 login`. If the organization does not already have
connector credentials, an organization admin creates them once:

```bash
ac2 harness credentials create
```

The command prints `AC2_SERVICE_CLIENT_ID` and `AC2_SERVICE_CLIENT_SECRET`. In a
dedicated second terminal, export those values and start the connector:

```bash
export AC2_SERVICE_CLIENT_ID=svc_...
export AC2_SERVICE_CLIENT_SECRET=...

uv run ac2 harness connect \
  --project-id <project-id> \
  --upstream http://127.0.0.1:8000
```

Keep both terminals running. The Harness API remains private; only the
connector makes an outbound connection to AC2. Run the connector with
`uv run` so it uses this project's SDK, which forwards grading; AC2 refuses to
launch a run without a grader until that connector is connected.

> Do not save these values with `ac2 config set` or put them in the project's
> `.env`. They are only for the Harness Connector process. Normal SDK commands
> and eval scripts should authenticate through `ac2 login`.

## Run the eval

In a third terminal without the connector environment variables:

```bash
uv run python -m swebench_opencode.eval
```

## Run training

With the Harness API and Harness Connector still running, start a training
smoke test:

```bash
uv run python -m swebench_opencode.train
```

The default is deliberately small: one Qwen3-4B training step, one problem per
batch, two samples, and a sampling concurrency of two. Pass `--steps`,
`--samples-per-problem`, or `--problem-batch-size` to run more.
