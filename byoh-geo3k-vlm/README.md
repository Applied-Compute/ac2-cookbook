# Train a VLM with your own harness

Train `Qwen/Qwen3.6-35B-A3B-VL` on Geometry3K using bring your own harness
(BYOH). Each task contains a geometry question and diagram. A small Python
harness sends them to AC2's Chat Completions relay, and a math grader scores
the final answer.

This example uses the [AC2-managed lifecycle](https://docs.appliedcompute.com/platform/custom-harnesses/ac2-managed):
AC2 runs your `CustomHarnessOrchestrator` inside the job and supplies a relay
URL and API key for each rollout. You do not deploy a Harness API or connector.
Replace `_complete` in `orchestrator.py` with your own harness call to adapt it.

## 1. Set up

Install [AC2](https://docs.appliedcompute.com/), then register the project in
your organization. Use an AC2 cluster with the 35B-A3B VLM and BYOH image
training enabled. The example requires AC2 SDK 0.3.46 or later.

From the cookbook root, run:

```bash
cd byoh-geo3k-vlm
export AC2_ORG_ID='YOUR_ORG_ID'
export AC2_PROJECT='byoh-geo3k-vlm'
ac2 project init
uv sync --locked
```

Replace `YOUR_ORG_ID` with your organization ID. Training uses AC2's managed
model; you do not need an OpenAI API key.

## 2. Upload a smoke dataset

Upload eight training tasks and eight validation tasks:

```bash
uv run python upload_dataset.py --smoke
```

The uploader reads a pinned revision of
[hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k), converts
diagrams to PNG data URLs, and preserves text and images as message content
parts. Expected answers go in `grader_params`, outside the model's input.
It creates `geo3k-smoke-train` and `geo3k-smoke-validation`.

## 3. Run two training steps

Push the project and submit to your cluster. Replace `YOUR_CLUSTER_ID` with
an AC2 cluster ID:

```bash
ac2 project push
uv run python -m byoh_geo3k_vlm.train --cluster YOUR_CLUSTER_ID --smoke
```

For a Modal cluster, add `--backend modal`:

```bash
uv run python -m byoh_geo3k_vlm.train --cluster YOUR_MODAL_CLUSTER_ID --backend modal --smoke
```

The config uses one training replica, one inference replica, and four samples
per problem. Replicas can span multiple GPUs. It evaluates before training and
after each smoke step. The `-VL` catalog suffix enables image training.

Follow the printed training ID:

```bash
ac2 train get TRAIN_ID
ac2 train logs TRAIN_ID
```

Inspect image traces, completion errors, and discarded samples in the
[AC2 platform](https://platform.appliedcompute.com/). A successful job status
alone does not establish rollout reliability. If image requests report
`Multimodal AC2 messages require a processor`, the cluster needs a training
runtime with BYOH image-processor support.

## 4. Train on the full dataset

After validating the smoke run, omit `--smoke` from both commands:

```bash
uv run python upload_dataset.py
uv run python -m byoh_geo3k_vlm.train --cluster YOUR_CLUSTER_ID
```

This uploads 2,101 training rows (2,100 unique tasks after deduplication) and
300 validation tasks into separate datasets, then runs 20 training steps with
evaluation every 10 steps. The test split remains unused. Add `--backend modal`
for a Modal cluster.

## How the harness works

`submit` starts one asynchronous completion and returns its ID. `get_status`
reports whether it is running, completed, or errored. `collect_outputs` returns
the answer and finish reason. AC2 records model calls through the relay and
passes that trace to `Geo3KGrader`, which compares the final assistant answer
in `\boxed{...}` with the expected answer. Teardown cancels remaining local
HTTP tasks.

The HTTP client sets `follow_redirects=True`. After 150 seconds,
[Modal can return a 303 result redirect](https://modal.com/docs/guide/webhook-timeouts).
Following that URL retrieves the same generation; it does not submit another
one. The harness does not automatically retry failed generation requests.

The completion deadline is 840 seconds across all redirects. The orchestrator
allows 900 seconds, and the training sample budget is 1,200 seconds to leave
room for trace collection and grading. For longer rollouts, update these
budgets together. Increasing the timeout alone does not enable redirects.

Run the offline checks with:

```bash
uv run pytest -q
```

See the [VLM guide](https://docs.appliedcompute.com/platform/training/vlm) for
image formats and the [BYOH overview](https://docs.appliedcompute.com/platform/custom-harnesses)
for other harness deployment models.
