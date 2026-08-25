# OpenAI Agents SDK web search through a self-hosted harness

This is the smallest complete example of the
[self-hosted BYOH deployment model](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted).
It is useful when you want to understand the three-endpoint Harness API before
connecting AC2 to a larger scheduler or sandbox system.

The example runs an OpenAI Agents SDK agent with a Wikipedia search tool. The
agent loop and tool run in a FastAPI process in your network. AC2 provides an
OpenAI-compatible model relay. The agent sends model requests to this endpoint;
AC2 routes them to the model selected for the job and records the trace. AC2
grades the answer returned by the API.

## What this example teaches

This is a **stateful, in-process harness**:

- `harness_server.py` starts each rollout as an `asyncio` task.
- The server keeps rollout IDs, statuses, and outputs in memory.
- `/collect_outputs` returns the answer and sources; the grader runs in AC2.

This keeps the lifecycle visible in one file. A production server can use the
same HTTP contract while storing state in a database and running the work in a
queue, cluster, or sandbox service.

## What you implement

[`harness_server.py`](harness_server.py) is the server you implement and
operate. Its FastAPI routes are the complete AC2 integration contract:

- `/submit` does not receive a rollout ID from AC2. The server validates the
  full task, creates an ID, starts `run_web_search`, and returns the ID to AC2.
  It uses `Idempotency-Key` to return the same ID if AC2 retries the request.
- `/get_status` reports the state of the corresponding `asyncio` task.
- `/collect_outputs` returns a cached `WebSearchOutput` for grading.

The server passes the rollout-scoped `ac_api_url` and `ac_api_key` from
`/submit` to an OpenAI-compatible client. That is how the agent uses the model
selected by AC2 and how AC2 records the trace.

The AC2 Harness Connector does not contain your harness logic. It is an
unmodified bridge that opens an outbound WebSocket to AC2 and forwards
requests to these private HTTP endpoints.

## How a rollout moves through the example

1. `eval.py` or `train.py` starts an AC2 job for a dataset task.
2. The AC2 job sends `/submit` through the Harness Gateway and Connector.
3. `harness_server.py` runs the agent and its Wikipedia tool.
4. The Connector polls `/get_status` locally and reports the terminal event.
5. AC2 requests `/collect_outputs`.
6. `grader.py` reads the returned answer and sources and computes the score in
   the AC2 job.

Your rollout code stays in your network; it does not run inside the AC2 job.

## Code map

| File | Responsibility |
| --- | --- |
| `harness_server.py` | Your Harness API, agent loop, tool, and in-memory rollout state |
| `src/byoh_agents_web_search_api/models.py` | JSON output contract shared by the server and grader |
| `src/byoh_agents_web_search_api/grader.py` | AC2-side grading of the returned answer |
| `src/byoh_agents_web_search_api/eval.py` | Eval configuration that selects the self-hosted lifecycle |
| `src/byoh_agents_web_search_api/train.py` | One-step connector-backed GRPO training configuration |
| `upload_dataset.py` | Small public dataset used by the example |

## Adapt it to your harness

For your own use case:

1. Replace `run_web_search` with the call that starts your agent, worker, or
   scheduler job.
2. Validate the task fields and `env_params` your harness needs.
3. Replace `WebSearchOutput` with your result schema.
4. Update `grader.py` to grade those outputs.
5. Move the idempotency, rollout, and output maps to durable storage before
   running multiple API replicas or relying on restart recovery.

The `/submit`, `/get_status`, and `/collect_outputs` routes stay the same.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

From this directory, register the project and install both AC2-side and local
Harness API dependencies:

```bash
ac2 project init
uv sync --extra harness
uv run python upload_dataset.py
```

BYOH is currently in beta. Contact the AC team to enable it for this project.
We will provision the required model relay and project configuration.

Store the model provider key in the project:

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
uv run python -m byoh_agents_web_search_api.eval
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
uv run python -m byoh_agents_web_search_api.train
```

The default configuration runs one GRPO step with two customer-hosted
rollouts. Pass `--steps` to run more steps.
