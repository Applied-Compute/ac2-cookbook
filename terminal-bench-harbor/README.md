# Terminal-Bench through Harbor on a self-hosted harness

[Terminal-Bench 2.0](https://www.tbench.ai/) as a
[self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted)
whose API is a front for an execution system you already run. For each task the
harness starts a [Harbor](https://github.com/harbor-framework/harbor) job;
Harbor runs OpenCode in a Modal sandbox and then runs the task's tests. AC2
supplies the model endpoint for OpenCode, records the trace, and stores the
score your harness returns.

The harness also grades: the score is the reward Harbor's verifier writes. If
the agent process fails, for example because the model endpoint rejects its
calls, the harness reports the rollout as failed with no output and no score. A
trial that ends at the task's agent timeout keeps its reward.

You need the AC2 CLI, Harness Connector credentials, a
[Modal](https://modal.com) account, and an OpenAI API key for the eval's model.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync --extra harness
uv run modal setup
uv run --extra harness python upload_dataset.py
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

`modal setup` opens a browser to log in and saves a Modal token on your
machine. `upload_dataset.py` downloads the tasks through Harbor and creates the
`terminal-bench` dataset with all 89 tasks and `terminal-bench-smoke` with
three short ones (`fix-git`, `openssl-selfsigned-cert`, and `regex-log`). The
eval job calls the eval's model, `gpt-5.6-luna`, so the OpenAI key is stored in
AC2; the sandbox only receives a key scoped to its rollout.

## Start your services

Start the Harness API on `http://127.0.0.1:8000`:

```bash
uv run --extra harness python harness_server.py
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

`<project-id>` is the `prj_...` ID in your project's URL on the platform.

Add `--transport https` if your network only allows plain HTTPS. Keep both
running.

## Run the eval

In a third terminal, without the connector variables:

```bash
uv run python -m terminal_bench_harbor.eval
```

It runs the three tasks in `terminal-bench-smoke` on `gpt-5.6-luna` through
your harness and prints the eval ID. Check the result with
`ac2 eval get <eval-id>`. Each task starts its own Modal sandbox and gives the
agent 15 minutes. Pass `--model` for another model.

Pass `--dataset terminal-bench` to run all 89 tasks. Terminal-Bench sets each
task's agent timeout, and some tasks allow the agent two hours or more. AC2
waits two hours for a rollout and then records it as an error. The harness
contract has no cancel call, so the Harbor trial and its Modal sandbox keep
running until Harbor's own timeout.

The example stops at eval: Terminal-Bench task files carry a canary string
asking that benchmark data never appear in training corpora.

## How it works

[`harness_server.py`](harness_server.py) is the server you operate. Its routes
are the whole AC2 integration:

- `/submit` receives the task and a model URL and key scoped to the rollout. It
  starts a Harbor job for the one Terminal-Bench task named in the task's
  `env_params`, creates a rollout ID, and returns it. `Idempotency-Key` makes a
  retried request return the same ID.
- `/get_status` reports `running` until the Harbor job finishes, then
  `completed`, or `failed` with Harbor's error message.
- `/collect_outputs` returns Harbor's job and trial IDs and the verifier
  rewards, or `null` for a trial with no usable reward. Harbor runs the tests
  even when the trial reports an error; the harness discards that reward unless
  the error is the agent timeout.
- `/submit_grading`, `/get_grading_status`, and `/collect_score` are the grading
  half. Harbor has already run the tests by the time the rollout finishes, so
  grading completes at once and `/collect_score` returns the verifier reward. A
  trial with no usable reward fails grading.

`_run_harbor` configures Harbor's `opencode` agent with one OpenCode provider
whose `baseURL` is the URL from `/submit` and whose key is read from the
sandbox environment. OpenCode's model calls therefore go to the model selected
for the job, and AC2 records them even though the sandbox is outside the AC2
job.

The server keeps its Harbor jobs and outputs in memory. To run more than one
replica or survive a restart, store the idempotency keys, the mapping from
rollout ID to job ID, and the outputs in a database.

To put your own execution system behind the same routes, replace `_run_harbor`
with the call that submits work to your scheduler, `HarborDataset` with the
task settings your server needs, and `RolloutOutput` with what you want
recorded.

`upload_dataset.py` gives each AC2 task the task instruction as its input and
the Harbor dataset, the dataset version's content hash, and the task name as
`env_params`.

## Source and license

The tasks are Harbor's
[`terminal-bench/terminal-bench-2`](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2)
dataset. Terminal-Bench 2.0 is released under the
[Apache License 2.0](https://github.com/harbor-framework/terminal-bench-2/blob/main/LICENSE).
