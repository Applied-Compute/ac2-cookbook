# Minimal self-hosted harness

The smallest Harness API that does everything a real one does. Each rollout
makes one model call through the URL and key AC2 sends with `/submit`, so the
rollout gets a trace and a graded answer, but there are no tools and no agent
framework. The only outbound traffic is that model call.

Use it to prove the [self-hosted BYOH](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted)
path end to end, on a laptop or in your cluster, before connecting your own
agent. To move to your agent, replace `run_rollout` in
[`harness_server.py`](harness_server.py) and keep the three routes.

## Set up the project

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync --extra harness
uv run python upload_dataset.py
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

The eval's default model, `gpt-5-mini`, is called from the eval job, so the key
is needed wherever the harness runs.

## Run it

Start the Harness API:

```bash
uv run --extra harness python harness_server.py
```

In a second terminal, start the connector with the organization's connector
credentials (an admin creates them once with `ac2 harness credentials create`):

```bash
export AC2_SERVICE_CLIENT_ID=svc_...
export AC2_SERVICE_CLIENT_SECRET=...

ac2 harness connect \
  --project-id <project-id> \
  --upstream http://127.0.0.1:8000 \
  --transport https
```

In a third terminal, without the connector variables:

```bash
uv run python -m byoh_minimal_api.eval
```

## Run it as a container

The `Dockerfile` builds the Harness API into an image that listens on
`0.0.0.0:8000` and runs as a non-root user. The build reads `ac2` from the AC2
package index, so pass your API key as a build secret:

```bash
AC2_API_KEY=key_... docker build --secret id=ac2_api_key,env=AC2_API_KEY \
  -t byoh-minimal-api .
```
