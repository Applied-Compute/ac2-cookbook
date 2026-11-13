# Terminal-Bench through Harbor on a self-hosted harness

This example shows the
[self-hosted BYOH deployment model](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted)
when the Harness API is a facade over an existing execution system rather than
the place where the agent itself runs.

The API runs [Terminal-Bench 2.0](https://www.tbench.ai/) tasks with OpenCode in
Modal sandboxes through [Harbor](https://github.com/harbor-framework/harbor),
the harness Terminal-Bench ships with. Harbor, Modal, the coding agent, and the
verifier all run in your infrastructure. AC2 connects the agent to the model
selected for the job, records the trace, and accepts the verifier reward
returned by the API.

The tasks are Harbor's
[`terminal-bench/terminal-bench-2`](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2)
dataset: 89 tasks, each with its own container image, instruction, and tests.
Terminal-Bench 2.0 is released under the
[Apache License 2.0](https://github.com/harbor-framework/terminal-bench-2/blob/main/LICENSE).
The example stops at eval: Terminal-Bench task files carry a canary string
asking that benchmark data never appear in training corpora.

You need an AC2 account, a [Modal](https://modal.com) account, and an OpenAI
API key. Harbor is installed with the example's dependencies, and Harbor
installs OpenCode inside each sandbox. The default eval runs three short tasks.

## What this example teaches

This is an **external execution and grading in your infrastructure** pattern:

- The Harness API coordinates a Harbor job instead of executing shell tools in
  the FastAPI process.
- Harbor and Modal own the task sandbox and agent lifecycle.
- Harbor's verifier computes the reward in your infrastructure.
- `/collect_score` returns that reward as the AC2 score. There is no AC2-side
  grader.

The example keeps the Harbor job handles and outputs in memory for readability.
In production, store the mapping from rollout ID to scheduler job ID in a
database. The API replicas can then be stateless while Harbor, Ray, LSF, or
another scheduler remains the source of truth.

## What you implement

[`harness_server.py`](harness_server.py) is the server you implement and
operate. Its FastAPI routes are the complete AC2 integration contract:

- `/submit` does not receive a rollout ID from AC2. The server validates the
  Harbor dataset settings, creates an ID, starts a job, and returns the ID to
  AC2. It uses `Idempotency-Key` to return the same ID if AC2 retries the
  request.
- `/get_status` translates Harbor job state into the BYOH status contract and
  passes on Harbor's error message for a failed trial.
- `/collect_outputs` caches and returns Harbor's job IDs, verifier rewards, and
  error metadata. It returns `null` for a trial that produced no reward.
  Harbor runs the tests even when the trial reports an error, for example
  because the model endpoint was down. Unless the error is the agent running
  out of time, the server discards that reward, so the trial fails instead of
  scoring 0.
- `/submit_grading`, `/get_grading_status`, and `/collect_score` are the grading
  half of the contract. Harbor has already run the task's tests by the time the
  rollout finishes, so grading completes at once and `/collect_score` returns
  the verifier reward. A trial that produced no reward fails grading instead of
  being scored.

The server gives OpenCode the rollout-scoped `ac_api_url` and `ac_api_key` from
`/submit`. OpenCode therefore uses the model selected by AC2, and AC2 records
its calls even though the sandbox is outside the AC2 job.

`PinnedAcpAgent` is not part of the contract. Harbor drives OpenCode through
an ACP client that it installs in the sandbox, and Harbor 0.20 does not run
with version 1.0 of that client, so the class keeps the client below 1.0.

The AC2 Harness Connector does not contain your harness logic. It is an
unmodified bridge that opens an outbound WebSocket to AC2 and forwards
requests to these private HTTP endpoints.

## How a rollout moves through the example

1. `eval.py` starts an AC2 eval over Terminal-Bench tasks.
2. The AC2 job sends `/submit` through the Harness Gateway and Connector.
3. `harness_server.py` starts a Harbor job in Modal for the one Terminal-Bench
   task named in the request.
4. Harbor launches OpenCode, which uses the model endpoint supplied by AC2.
5. Harbor runs the task's tests after the agent finishes.
6. The Connector reports the terminal status and AC2 requests
   `/collect_outputs`.
7. AC2 requests grading and `/collect_score` returns the Harbor reward as the
   AC2 score.

Harbor and your rollout code stay in your network; they do not run inside the
AC2 job.

## Code map

| File | Responsibility |
| --- | --- |
| `harness_server.py` | Your Harness API and Harbor job coordinator |
| `src/terminal_bench_harbor/eval.py` | Eval configuration that selects the self-hosted lifecycle |
| `upload_dataset.py` | Downloads the tasks through Harbor and creates one AC2 task per Terminal-Bench task |

## Adapt it to your harness

For your own use case:

1. Replace `_run_harbor` and `PinnedAcpAgent` with the call that submits work
   to your scheduler or sandbox system.
2. Replace `HarborDataset` with the private task IDs and execution settings your
   server needs.
3. Translate your scheduler's states in `/get_status`.
4. Replace `RolloutOutput` with the artifacts and metadata you want
   recorded, and return your own score from `/collect_score`.
5. Persist idempotency keys, job IDs, and cached outputs before running multiple
   replicas or relying on restart recovery.

The routes stay the same.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

From this directory, register the project and install both AC2-side and local
Harness API dependencies:

```bash
ac2 project init
uv sync --extra harness
uv run modal setup
uv run --extra harness python upload_dataset.py
```

`modal setup` opens a browser to log in and saves a Modal token on your
machine; it stays there.

`upload_dataset.py` uploads all 89 tasks to the `terminal-bench` dataset and
three short ones (`fix-git`, `openssl-selfsigned-cert`, and `regex-log`) to
`terminal-bench-smoke`. Each AC2 task carries the task instruction as its input
and the Harbor dataset, the dataset version's content hash, and task name as
`env_params`.

AC2 calls the model, so the model provider key is stored in AC2. The sandbox
only receives a key scoped to its rollout:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

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
connector makes an outbound connection to AC2.

By default the connector holds one outbound WebSocket to AC2. If your network
only allows plain HTTPS requests, run it in HTTPS mode instead: it then fetches
work with a long poll and posts results, and every connection it makes is an
HTTPS request that returns within `--poll-wait` seconds (30 by default, 120 at
most).

```bash
uv run ac2 harness connect \
  --project-id <project-id> \
  --upstream http://127.0.0.1:8000 \
  --transport https \
  --poll-wait 30
```

Stop the connector with Ctrl-C or SIGTERM so it releases the project at once;
a connector that is killed outright is replaced after about a minute.

> Do not save these values with `ac2 config set` or put them in the project's
> `.env`. They are only for the Harness Connector process. Normal SDK commands
> and eval scripts should authenticate through `ac2 login`.

## Run the eval

In a third terminal without the connector environment variables:

```bash
uv run python -m terminal_bench_harbor.eval
```

The eval runs the three tasks in `terminal-bench-smoke` with `gpt-5.6-luna`.
Each starts its own Modal sandbox and has a 15-minute agent timeout.

Pass `--dataset terminal-bench` to run all 89 tasks. Terminal-Bench sets each
task's agent timeout, and AC2 waits two hours for a rollout before recording it
as an error. Two tasks allow the agent two hours or more.

The harness contract has no cancel call. After AC2 gives up on a rollout or the
eval is interrupted, the Harbor trial keeps running, and its Modal sandbox
keeps billing, until Harbor's own task timeout.

If connector credentials were previously saved globally, remove them first:

```bash
ac2 config unset service_client_id
ac2 config unset service_client_secret
ac2 login
```
