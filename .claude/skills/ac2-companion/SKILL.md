---
name: ac2-companion
description: Helps users connect their own agent harness to Applied Compute Agent Cloud (AC2) and run evals and training through it. Use when the user mentions AC2, the `ac2` CLI, a self-hosted or custom harness (BYOH), the Harness API (`/submit`, `/get_status`, `/collect_outputs`), the Harness Connector (`ac2 harness connect`), `CustomHarnessConfig`, `EvalConfig`, `TrainingConfig`, or `platform.appliedcompute.com`.
argument-hint: "[task or question]"
allowed-tools: Bash(ac2 config show*) Bash(ac2 harness status*) Bash(ac2 eval get*) Bash(ac2 eval list*) Bash(ac2 eval logs*) Bash(ac2 train get*) Bash(ac2 train list*) Bash(ac2 train logs*) Bash(ac2 train checkpoints*) Bash(ac2 datasets list*) Bash(ac2 secrets list*)
user-invocable: true
---

# AC2 companion

Help the user run evals and training on AC2 with a self-hosted harness: their
agent, tools, and grading run on their infrastructure, and AC2 serves the model
and runs the job.

Docs: <https://docs.appliedcompute.com/platform/custom-harnesses> and
<https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted>. The
index of all pages is <https://docs.appliedcompute.com/llms.txt>. When this file
and the docs or `ac2 <command> --help` disagree, trust those. The exception is
grading in the harness, which the docs do not cover: follow this file and
`ac2.harness.types`.

## The pieces

- **Harness API**: an HTTP server the user writes and runs. Start from
  `byoh-smoke-example/harness_server.py`.
- **Harness Connector**: `ac2 harness connect`. It connects out to AC2 and
  forwards requests to the Harness API, so the server needs no inbound access.
- **AC2 job**: an eval or training run started with the SDK. It sends each task
  to the harness and serves the model the agent calls.

## Harness API

All endpoints are `POST` with JSON bodies. Request and response models are in
`ac2.harness.types`.

| Endpoint | Request | Response |
| --- | --- | --- |
| `/submit` | `ac_api_url`, `ac_api_key`, `task.input`, optional `task.env_params` | `rollout_id` |
| `/get_status` | `rollout_id` | `status`, optional `error_message`, `annotations`, `spans` |
| `/collect_outputs` | `rollout_id` | Any JSON, up to 1 MiB |

- `/submit` starts the work and returns at once. The server creates the
  `rollout_id`. AC2 sends an `Idempotency-Key` header; return the same ID when a
  key repeats.
- The agent makes its model calls with an OpenAI-compatible client pointed at
  `ac_api_url` with `ac_api_key`. That is how it reaches the model selected for
  the job and how AC2 records the trace.
- `status` is one of `queued`, `running`, `completed`, `failed`, `cancelled`,
  `truncated`.

### Grading in the harness

When the eval or training config has no grader, AC2 asks the harness for the
score through three more endpoints:

| Endpoint | Request | Response |
| --- | --- | --- |
| `/submit_grading` | `rollout_id`, `rollout_status`, `task.input`, optional `task.env_params` and `task.grader_params` | Ignored |
| `/get_grading_status` | `rollout_id` | `status`, optional `error_message`, `annotations`, `spans` |
| `/collect_score` | `rollout_id` | `score`, optional `reasoning`, `artifacts` |

Grading `status` is one of `queued`, `running`, `completed`, `failed`,
`cancelled`. These endpoints need `ac2` 0.3.97 or later in the project
(`uv lock --upgrade-package ac2 && uv sync`) and a current CLI running the
connector (`ac2 upgrade`).

- When the model endpoint or infrastructure failed, return `failed` with
  `error_message` from `/get_status`, `null` from `/collect_outputs`, and
  `failed` from `/get_grading_status`. Never return a score of 0 for an outage.
  Return 404 for a rollout ID the server does not know.

## Workflow

1. Install and log in: `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.
   Re-authenticate later with `ac2 login`. Logging in also gives `uv` access to
   the AC2 package index.
2. Register the project from its folder with `ac2 project init`, then install
   it with `uv sync` (add `--extra <name>` when the example's README does). If
   the sync fails on index auth, run `ac2 login` again. If it still fails, set
   `UV_INDEX_AC2_USERNAME=__token__` and `UV_INDEX_AC2_PASSWORD` to the user's
   AC2 API key, then retry.
3. Upload the dataset with the example's `upload_dataset.py`. It uses
   `client.datasets.create` and `client.datasets.add_tasks` with `Task` objects
   (`id`, `input`, `env_params`, `grader_params`).
4. Start the Harness API.
5. Start the connector in its own terminal. An organization admin creates its
   credentials once with `ac2 harness credentials create`.

   ```bash
   export AC2_SERVICE_CLIENT_ID=svc_...
   export AC2_SERVICE_CLIENT_SECRET=...
   ac2 harness connect --project-id <project-id> --upstream http://127.0.0.1:8000
   ```

   `<project-id>` is the `prj_...` ID in the project's URL on the platform.
   Add `--transport https` if the network only allows plain HTTPS. Check
   connections with `ac2 harness status`.
6. Run a baseline eval from another terminal, without the connector variables.
7. Run a short training job. Follow the example's README.

Eval and training configs select the connector by setting `custom_harness`
(`CustomHarnessConfig` in `EvalConfig`, `TrainingCustomHarnessConfig` in
`TrainingConfig`) and leaving out the agent, environment, orchestrator, and
grader. Copy `byoh-smoke-example/src/byoh_smoke_example/eval.py` and
`train.py`.

## Inspect a run

```bash
ac2 eval get <eval-id>
ac2 eval logs <eval-id>
ac2 train get <train-id>
ac2 train logs <train-id>
ac2 train checkpoints <train-id>
ac2 traces download <run-id>
```

## Rules

- Keep connector credentials in the connector's terminal only. Do not save them
  with `ac2 config set` or in `.env`; SDK commands authenticate through
  `ac2 login`. If they were saved, run `ac2 config unset service_client_id` and
  `ac2 config unset service_client_secret`, then `ac2 login`.
- One connector serves a project at a time.
- Model provider keys for an eval go in `ac2 secrets put --key <NAME> --value <value>`.
- Verify a command with `--help` before giving it to the user. Do not run
  commands that create credentials, store secrets, or start jobs unless asked.
