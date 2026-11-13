# Self-hosted harness smoke example

The smallest [self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted):
each rollout makes one model call through the URL and key AC2 sends with
`/submit`, so it gets a trace and a graded answer. Start here, then replace
`run_rollout` in [`harness_server.py`](harness_server.py) with your agent.

The harness also grades: a rollout scores 1 when its answer contains the task's
`expected_answer`, and 0 otherwise. If the model call fails, the harness reports
the rollout as failed with no output and no score, so an outage is not counted
as a wrong answer.

## Set up

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync --extra harness
uv run python upload_dataset.py
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

The eval's model, `gpt-5.6-luna`, is called from the eval job, so it needs the
key wherever the harness runs.

## Run

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
  --upstream http://127.0.0.1:8000 \
  --transport https
```

Run the eval in a third terminal, without the connector variables:

```bash
uv run python -m byoh_smoke_example.eval
```

## Train

With the Harness API and connector still running, start a one-step run:

```bash
uv run python -m byoh_smoke_example.train
```

It runs one training step with two rollouts through your harness. Training calls
the model being trained, so it needs no OpenAI key. Pass `--steps` for more.

## Build the image

The image serves the Harness API on port 8000 as a non-root user. Its build
installs `ac2` from the AC2 package index, so pass your API key as a secret:

```bash
AC2_API_KEY=key_... docker build --secret id=ac2_api_key,env=AC2_API_KEY \
  -t byoh-smoke-example .
```
