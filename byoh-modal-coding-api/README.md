# Modal coding agent through a self-hosted harness

This example shows the
[self-hosted BYOH deployment model](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted)
when the Harness API is a facade over an existing execution system rather than
the place where the agent itself runs.

The API starts Harbor's public
[`harbor/hello-world@latest`](https://hub.harborframework.com/datasets/harbor/hello-world/latest)
task with OpenCode in a Modal sandbox. Harbor, Modal, the coding agent, and the
verifier all run in your infrastructure. AC2 provides an OpenAI-compatible
model relay. The agent sends model requests to this endpoint; AC2 routes them
to the model selected for the job and records the trace. AC2 accepts the
verifier reward returned by the API.

## What this example teaches

This is an **external execution and grading in your infrastructure** pattern:

- The Harness API coordinates a Harbor job instead of executing shell tools in
  the FastAPI process.
- Harbor and Modal own the task sandbox and agent lifecycle.
- Harbor's verifier computes the reward in your infrastructure.
- `/collect_outputs` returns that reward, and the AC2 grader passes it through.

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
- `/get_status` translates Harbor job state into the BYOH status contract.
- `/collect_outputs` caches and returns Harbor's job IDs, verifier rewards, and
  error metadata.

The server gives OpenCode the rollout-scoped `ac_api_url` and `ac_api_key` from
`/submit`. OpenCode therefore uses the model selected by AC2, and AC2 records
its calls even though the sandbox is outside the AC2 job.

The AC2 Harness Connector does not contain your harness logic. It is an
unmodified bridge that opens an outbound WebSocket to AC2 and forwards
requests to these private HTTP endpoints.

## How a rollout moves through the example

1. `eval.py` or `train.py` starts an AC2 job for the Harbor task.
2. The AC2 job sends `/submit` through the Harness Gateway and Connector.
3. `harness_server.py` starts a Harbor job in Modal.
4. Harbor launches OpenCode, which calls the AC2 relay for model completions.
5. Harbor runs the task verifier after the agent finishes.
6. The Connector reports the terminal status and AC2 requests
   `/collect_outputs`.
7. `grader.py` reads the returned Harbor reward and passes it through as the
   AC2 score.

Harbor and your rollout code stay in your network; they do not run inside the
AC2 job.

## Code map

| File | Responsibility |
| --- | --- |
| `harness_server.py` | Your Harness API and Harbor job coordinator |
| `src/byoh_modal_coding_api/models.py` | Harbor task settings and JSON output contract |
| `src/byoh_modal_coding_api/grader.py` | AC2-side validation and pass-through of the reward computed in your infrastructure |
| `src/byoh_modal_coding_api/eval.py` | Eval configuration that selects the self-hosted lifecycle |
| `src/byoh_modal_coding_api/train.py` | One-step connector-backed GRPO training configuration |
| `upload_dataset.py` | Creates the AC2 dataset entry that identifies the Harbor task |

## Adapt it to your harness

For your own use case:

1. Replace `_run_harbor` with the call that submits work to your scheduler or
   sandbox system.
2. Replace `HarborDataset` with the private task IDs and execution settings your
   server needs.
3. Translate your scheduler's states in `/get_status`.
4. Replace `CodingRolloutOutput` with the artifacts, score, and metadata your
   grader needs.
5. Persist idempotency keys, job IDs, and cached outputs before running multiple
   replicas or relying on restart recovery.

The `/submit`, `/get_status`, and `/collect_outputs` routes stay the same.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

From this directory, register the project and install both AC2-side and local
Harness API dependencies:

```bash
ac2 project init
uv sync --extra harness
uv run modal setup
uv run python upload_dataset.py
```

BYOH is currently in beta. Contact the AC team to enable it for this project.
We will provision the required model relay and project configuration.

Modal credentials remain on your machine. Store the model provider key
in the project:

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
ac2 harness credentials rotate
```

The command prints `AC2_SERVICE_CLIENT_ID` and `AC2_SERVICE_CLIENT_SECRET`. In a
dedicated second terminal, export those values and start the connector:

```bash
export AC2_SERVICE_CLIENT_ID=svc_...
export AC2_SERVICE_CLIENT_SECRET=...

ac2 harness connect \
  --project-id <project-id> \
  --upstream http://127.0.0.1:8000
```

Keep both terminals running. The Harness API remains private; only the
connector makes an outbound connection to AC2.

> Do not save these values with `ac2 config set` or put them in the project's
> `.env`. They are only for the Harness Connector process. Normal SDK commands
> and eval scripts should authenticate through `ac2 login`.

## Run the eval

In a third terminal without the connector environment variables:

```bash
uv run python -m byoh_modal_coding_api.eval
```

If connector credentials were previously saved globally, remove them first:

```bash
ac2 config unset service_client_id
ac2 config unset service_client_secret
ac2 login
```

## Run training

With the Harness API and Harness Connector still running, start a training
smoke test:

```bash
uv run python -m byoh_modal_coding_api.train
```

The default configuration runs one GRPO step with two customer-hosted Harbor
rollouts. Pass `--steps` to run more steps.
