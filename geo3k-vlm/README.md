# Train a VLM on Geometry3K with a self-hosted harness

Train a vision-language model on image questions through a
[self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted).
Each task is a geometry question with a diagram. The Harness API in
[`harness_server.py`](harness_server.py) sends the question and image to the
model endpoint AC2 supplies with `/submit`, then grades the LaTeX answer in
the response itself with `math-verify` and returns the score through
`/submit_grading`, `/get_grading_status` and `/collect_score`. No agent,
environment or grader code runs in the AC2 job.

## Set up

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync --extra harness
uv run python upload_dataset.py
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

The uploader reads a pinned revision of
[hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)
(MIT license, converted from [InterGPS](https://github.com/lupantech/InterGPS))
and creates `geo3k-train` (2,100 tasks) and `geo3k-validation` (300 tasks).
Questions and PNG diagrams become message content parts; expected answers go
in `grader_params`, which AC2 sends to the harness only for grading. The test
split is left unused.

The eval's model, `gpt-5.6-luna`, is called from the eval job, which reads the
OpenAI key from that AC2 secret. Training needs no OpenAI key.

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

Run the baseline eval in a third terminal, without the connector variables:

```bash
uv run python -m geo3k_vlm.eval
```

It runs 8 validation tasks through your harness with `gpt-5.6-luna`.

## Train

With the Harness API and connector still running:

```bash
uv run python -m geo3k_vlm.train
```

It trains `Qwen/Qwen3.5-9B-VL` for 20 steps of 8 tasks with 4 rollouts each.
The `-VL` suffix enables image training. Pass `--steps` for a shorter or longer
run.

`max_response_len=8192` caps each answer; the harness reports a rollout that
reaches the cap as `truncated`.

See the [VLM guide](https://docs.appliedcompute.com/platform/training/vlm)
for more on image inputs.
