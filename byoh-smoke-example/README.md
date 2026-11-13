# Self-hosted harness smoke example

The smallest [self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted).
Run it first to check your credentials, your network path to AC2, and the
Harness Connector, then replace `run_rollout` in
[`harness_server.py`](harness_server.py) with your agent.

Each rollout makes one model call through the URL and key AC2 sends with
`/submit`. The harness sends `policy` as the model name; AC2 serves the model
selected for the job and records the trace.

The harness also grades: a rollout scores 1 when its answer contains the task's
`expected_answer`, and 0 otherwise. If the model call fails, the harness reports
the rollout as failed with no output and no score, so an outage is not counted
as a wrong answer.

You need the AC2 CLI, Harness Connector credentials, and an OpenAI API key for
the eval's model.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync --extra harness
uv run python upload_dataset.py
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

`upload_dataset.py` creates the `byoh-smoke-eval` dataset with three factual
questions. The eval job calls the eval's model, `gpt-5.6-luna`, so the OpenAI
key is stored in AC2 and the harness does not need it.

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

`<project-id>` is the `prj_...` ID in your project's URL on the platform.

Add `--transport https` if your network only allows plain HTTPS. Keep both
running.

## Run the eval

In a third terminal, without the connector variables:

```bash
uv run python -m byoh_smoke_example.eval
```

It runs the three questions through your harness and prints the eval ID. Check
the result with `ac2 eval get <eval-id>`.

## Run training

With the Harness API and connector still running:

```bash
uv run python -m byoh_smoke_example.train
```

It runs one training step of `Qwen/Qwen3-4B` with two rollouts through your
harness and prints the training run ID. Check the run with
`ac2 train get <train-id>`. Training calls the model being trained, so it needs
no OpenAI key. `global_sampling_concurrency` in `train.py` caps how many
rollouts reach your harness at once. Pass `--steps` for more steps.
