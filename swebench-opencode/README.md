# OpenCode on SWE-bench through a self-hosted harness

A [self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted)
that runs the unmodified OpenCode CLI on
[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
tasks and grades each patch with the official SWE-bench tests. OpenCode, the
sandboxes, and the grading run in your infrastructure. AC2 serves the model
OpenCode calls and records the trace and the score.

Use it as the starting point for a coding agent that needs a sandbox per
rollout: in [`harness_server.py`](harness_server.py), replace the OpenCode
command in `run_rollout` with your agent and `grade` with your scoring.

## What you need

- An AC2 account and the `ac2` CLI.
- A [Modal](https://modal.com) account. The sandboxes run there and Modal bills
  you for them: each rollout uses one sandbox for OpenCode and then one for the
  tests, with 4 CPUs and 16 GB of memory each. They appear in your Modal
  workspace under the app `swebench-opencode`.
- An OpenAI API key for the eval, which uses `gpt-5.6-luna`. Training calls the
  model being trained and needs no key.
- Harness Connector credentials, which an organization admin creates once.

## Set up

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

From this directory:

```bash
ac2 project init
uv sync --extra harness
uv run modal setup
uv run python upload_dataset.py
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

`modal setup` stores a Modal token on your machine; it is never sent to AC2.
`upload_dataset.py` uploads the first two SWE-bench Verified tasks to the
`swebench-opencode-smoke` dataset. Pass `--num-tasks` for more, or
`--instance-id <id>` one or more times to choose tasks.

## Run

Start the Harness API on `http://127.0.0.1:8000`:

```bash
uv run --extra harness uvicorn harness_server:app
```

Start the connector in a second terminal. An organization admin creates its
credentials once with `ac2 harness credentials create`, which prints the two
values to export. `<project-id>` is your project's `prj_...` ID.

```bash
export AC2_SERVICE_CLIENT_ID=svc_...
export AC2_SERVICE_CLIENT_SECRET=...

uv run ac2 harness connect \
  --project-id <project-id> \
  --upstream http://127.0.0.1:8000
```

The connector makes an outbound connection to AC2 and forwards requests to the
Harness API, which stays private. Keep the credentials out of `ac2 config set`
and the project's `.env`: they are for the connector process only.

Run the eval in a third terminal, without the connector variables:

```bash
uv run python -m swebench_opencode.eval
```

It prints the eval ID and runs the two tasks one at a time. The first rollout
of a task is slower because Modal builds an image from the task's public
SWE-bench image.

## Train

With the Harness API and connector still running, start a one-step run:

```bash
uv run python -m swebench_opencode.train
```

It runs one Qwen3-4B training step with two rollouts of one task, which checks
the training path end to end. Pass `--steps`, `--samples-per-problem`, or
`--problem-batch-size` to run more; every rollout in a step runs at once, each
in its own sandbox.

## How it works

`harness_server.py` is the whole integration. AC2 calls its six routes through
the connector:

- `/submit` starts a rollout and returns its ID. The rollout starts a Modal
  sandbox from the task's SWE-bench image, writes the issue to
  `/task/instruction.md`, and runs OpenCode against the `ac_api_url` and
  `ac_api_key` AC2 sent. When OpenCode exits, the server diffs the repository
  against the image's commit and terminates the sandbox. `Idempotency-Key`
  returns the same ID if AC2 retries the request.
- `/get_status` reports `running`, `completed`, or `failed`.
- `/collect_outputs` returns the sandbox ID, the patch size, the exit code, and
  the tail of OpenCode's output.
- `/submit_grading` applies the patch in a clean sandbox and runs the official
  tests.
- `/get_grading_status` reports `running`, `completed`, or `failed`.
- `/collect_score` returns 1 if the task is resolved and 0 otherwise, with the
  test results.

A rollout that leaves no patch because OpenCode reported an error or exited
non-zero is `failed`, with no outputs and no score, so a model or sandbox
outage is not counted as a wrong answer. A rollout that exits non-zero but
leaves a patch is `failed` and still graded on that patch. A patch that does
not apply scores 0.

OpenCode is stopped after 90 minutes. The tests are stopped after 25 minutes,
inside the 30 minutes AC2 waits for a grading by default
(`grading_timeout_seconds` on the harness config).

Rollouts and gradings are kept in memory. Store them in a database before you
restart the server mid-run or run more than one replica.

## Files

| File | Responsibility |
| --- | --- |
| `harness_server.py` | The Harness API: OpenCode sandboxes, the patch, and SWE-bench grading |
| `upload_dataset.py` | Creates the AC2 dataset from SWE-bench Verified |
| `src/swebench_opencode/eval.py` | Starts the eval |
| `src/swebench_opencode/train.py` | Starts the training run |
