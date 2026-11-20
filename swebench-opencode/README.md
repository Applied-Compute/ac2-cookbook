# OpenCode on SWE-bench through a self-hosted harness

A [self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted)
that runs the unmodified OpenCode CLI on
[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
tasks. Each rollout starts a Modal sandbox from the task's SWE-bench image, runs
OpenCode in it, and takes the git diff when OpenCode exits. AC2 serves the model
OpenCode calls and records the trace.

The harness also grades: it applies the patch in a second sandbox and runs the
official SWE-bench tests. A rollout scores 1 when the task is resolved, and 0
otherwise. If OpenCode fails and leaves no patch, the harness reports the
rollout as failed with no output and no score, so an outage is not counted as a
wrong answer.

Use it as the starting point for a coding agent that needs a sandbox per
rollout: in [`harness_server.py`](harness_server.py), replace the OpenCode
command in `run_rollout` with your agent and `grade` with your scoring.

You need the AC2 CLI, Harness Connector credentials, and:

- A [Modal](https://modal.com) account. The sandboxes run there and Modal bills
  you for them: each rollout uses one sandbox for OpenCode and then one for the
  tests, with 4 CPUs and 16 GB of memory each. They appear in your Modal
  workspace under the app `swebench-opencode`.
- An OpenAI API key for the eval's model, `gpt-5.6-luna`. Training calls the
  model being trained and needs no key.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

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

## Start your services

Start the Harness API on `http://127.0.0.1:8000`:

```bash
uv run --extra harness uvicorn harness_server:app
```

Start the connector in a second terminal. An organization admin creates its
credentials once with `ac2 harness credentials create`.

```bash
export AC2_SERVICE_CLIENT_ID=svc_...
export AC2_SERVICE_CLIENT_SECRET=...

ac2 harness connect \
  --project-id <project-id> \
  --upstream http://127.0.0.1:8000
```

`<project-id>` is the `prj_...` ID in your project's URL on the platform. Add
`--transport https` if your network only allows plain HTTPS. Keep both running.

## Run the eval

In a third terminal, without the connector variables:

```bash
uv run python -m swebench_opencode.eval
```

It runs the two tasks through your harness, one at a time, and prints the eval
ID. Check the result with `ac2 eval get <eval-id>`. The first rollout of a task
is slower because Modal builds an image from the task's public SWE-bench image.

## Run training

With the Harness API and connector still running:

```bash
uv run python -m swebench_opencode.train
```

It runs one training step of `Qwen/Qwen3-4B` with two rollouts of one task
through your harness and prints the training run ID. Check the run with
`ac2 train get <train-id>`. Pass `--steps`, `--samples-per-problem`, or
`--problem-batch-size` to run more. `global_sampling_concurrency` is set to the
number of rollouts in a step, so they all reach your harness at once, each in
its own sandbox.

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

A rollout is `failed` when OpenCode's last event is an error or it exits
non-zero. If it left a patch, the patch is still returned and graded. If it
left none, the rollout has no outputs and no score.

Before applying a patch, grading checks that its sandbox is a clean checkout of
the commit the rollout started from. A patch that does not apply scores 0 as an
invalid submission, with the `git apply` error in the score's artifacts. Tests
that are still running at the time limit score 0, as SWE-bench counts them. A
sandbox error or test output that cannot be parsed fails the grading and gives
no score.

OpenCode is stopped after 90 minutes. The tests are stopped after 25 minutes,
inside the 30 minutes AC2 waits for a grading by default
(`grading_timeout_seconds` on the harness config).

Rollouts and gradings are kept in memory. Store them in a database before you
restart the server mid-run or run more than one replica.
