# Train a VLM on Geometry3K with a self-hosted harness

Train a vision-language model on image questions through a
[self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted).
Each task is a geometry question with a diagram. The Harness API in
[`harness_server.py`](harness_server.py) makes one model call per question and
grades the LaTeX answer itself with `math-verify`: 1 when the final answer
matches the expected one, 0 otherwise.

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

`upload_dataset.py` reads a pinned revision of
[hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) and
creates `geo3k-train` (2,100 tasks) and `geo3k-validation` (300 tasks). The
upload is about 60 MB. The eval job calls the eval's model, `gpt-5.6-luna`, so
the OpenAI key is stored in AC2 and the harness does not need it.

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
uv run python -m geo3k_vlm.eval
```

It runs 8 validation tasks through your harness with `gpt-5.6-luna` and prints
the eval ID. Check the result with `ac2 eval get <eval-id>`.

## Run training

With the Harness API and connector still running:

```bash
uv run python -m geo3k_vlm.train
```

It trains `Qwen/Qwen3.5-9B-VL` for one step of 8 tasks with 4 rollouts each and
prints the training run ID. Check the result with `ac2 train get <train-id>`.
Training calls the model being trained, so it needs no OpenAI key. Pass
`--steps` for more.

## How it works

Each task's input is one user message holding the question text and its PNG
diagram as content parts. The expected answer is in `grader_params`, which AC2
sends to the harness only for grading.

AC2 sends each task to `/submit` with a model URL and key. The harness sends
the question and image to that endpoint with `policy` as the model name; AC2
serves the model selected for the job and records the trace. AC2 then asks the
harness to grade through `/submit_grading`, `/get_grading_status` and
`/collect_score`.

If the model call fails, the harness reports the rollout as failed with no
output and no score, so an outage is not counted as a wrong answer.

In [`train.py`](src/geo3k_vlm/train.py), the `-VL` suffix on the model name
enables image training. `max_response_len=8192` caps each response; the harness
reports a rollout that reaches the cap as `truncated` and scores it 0.
`global_sampling_concurrency=32` is the most rollouts AC2 sends to your harness
at once. See the [VLM guide](https://docs.appliedcompute.com/platform/training/vlm)
for more on image inputs.

## Source and license

Geometry3K comes from [InterGPS](https://github.com/lupantech/InterGPS) (MIT
license), read here through
[hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k). The
test split is not used.
