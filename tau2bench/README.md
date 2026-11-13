# tau2-bench through a self-hosted harness

The [tau2-bench](https://github.com/sierra-research/tau2-bench) customer-service
benchmark as a
[self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted).
The whole benchmark runs in a FastAPI process in your network: the conversation
loop, the domain tools and database, the simulated customer, and the tau2
reward. AC2 supplies the model endpoint for the agent, records the trace, and
stores the score your harness returns.

You need the AC2 CLI, Harness Connector credentials, and an OpenAI API key. The
key is used in two places: AC2 stores it as a secret so the eval job can call
the agent's model, and your harness reads it from the environment for the
simulated customer and the judge.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync --extra harness
uv run python upload_dataset.py --domain airline
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

`upload_dataset.py` creates `tau2bench-airline-train` (30 tasks),
`tau2bench-airline-test` (20), and `tau2bench-airline-base` (all 50). The
secret is for the eval's model, `gpt-5.6-luna`.

## Start your services

Start the Harness API on `http://127.0.0.1:8000` with your OpenAI key in its
environment:

```bash
export OPENAI_API_KEY=<openai-api-key>
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
uv run python -m tau2bench.eval
```

It runs the 20 airline test tasks on `gpt-5.6-luna` through your harness and
prints the eval ID. Check the result with `ac2 eval get <eval-id>`. Pass
`--model` for another model. For the retail domain, upload it with
`upload_dataset.py --domain retail` and pass `--dataset tau2bench-retail-test`.

## Run training

With the Harness API and connector still running:

```bash
uv run python -m tau2bench.train
```

It runs one training step of `Qwen/Qwen3-4B` on two airline tasks with four
conversations each and prints the training run ID. Check the run with
`ac2 train get <train-id>`. Training calls the model being trained, so it does
not use the stored secret; the harness still uses your key for the simulated
customer. `global_sampling_concurrency` in
[`train.py`](src/tau2bench/train.py) bounds how many conversations reach your
harness at once. Pass `--steps` for more steps. The run saves a checkpoint when
it finishes.

## How it works

This is a multi-turn harness that also grades:

- Each rollout is an `asyncio` task that alternates between a simulated
  customer and the agent until the customer ends the conversation or the agent
  transfers to a human. A conversation that passes 200 messages, or in which
  the agent sends an empty reply, is cut short and scores 0.
- The agent's model calls go to the `ac_api_url` and `ac_api_key` AC2 sends
  with `/submit`, using the model name `policy`. The harness executes every
  tool call against an in-memory copy of the domain database.
- The simulated customer calls `gpt-5.6-luna` with your own OpenAI key, and
  so does the natural-language judge for tasks whose reward includes
  `NL_ASSERTION`. Those calls are not part of the AC2 trace.
- The eval and training configs pass no grader. AC2 asks the harness for the
  score, and the harness compares the final database with a replay of the
  task's expected actions and checks the required information was
  communicated.

[`harness_server.py`](harness_server.py) keeps rollouts in memory and exposes
six routes:

| Route | What it does |
| --- | --- |
| `/submit` | Reads `domain`, `user_instructions`, and `initial_state` from the task's `env_params`, starts the conversation, and returns a rollout ID. `Idempotency-Key` makes a retry return the same ID. |
| `/get_status` | Reports whether the conversation is running, completed, or failed. |
| `/collect_outputs` | Returns the message count and, when the conversation was cut short, the reason. Returns `null` when the conversation failed. |
| `/submit_grading` | Receives the task's `grader_params` and starts grading the finished conversation. Grading a failed conversation fails, so a model or network outage is not scored. |
| `/get_grading_status` | Reports whether grading is running, completed, or failed. |
| `/collect_score` | Returns the score, the reasoning, and the per-component breakdown. |

The Harness Connector is an unmodified bridge: it makes outbound requests to
AC2 and forwards them to these private routes.

| File | Responsibility |
| --- | --- |
| `harness_server.py` | Harness API, conversation loop, and rollout state |
| `src/tau2bench/environments/` | Domain tools and databases |
| `src/tau2bench/user.py` | Simulated customer |
| `src/tau2bench/graders/` | tau2 reward |
| `src/tau2bench/dataloader/` | Fetches tau2-bench tasks, policies, and databases |
| `src/tau2bench/eval.py` | Eval configuration |
| `src/tau2bench/train.py` | Training configuration |
| `upload_dataset.py` | Registers the tau2-bench splits as AC2 datasets |

`.ac2ignore` keeps the harness and benchmark code out of the project AC2
uploads.

## Source and license

Tasks, policies, and databases are downloaded from a pinned commit of
[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
(MIT License, Copyright (c) 2025 Sierra Research) into
`src/tau2bench/dataloader/cache/` on first use. The domain tools, customer
simulator prompt, and reward follow that repository.
