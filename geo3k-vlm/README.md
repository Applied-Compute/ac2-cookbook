# Train a VLM on Geometry3K

Train a vision-language model through AC2 Dispatch using the model catalog.
Each task contains a geometry question and diagram. The agent answers in
`\boxed{...}`, and a math grader scores the answer.

## 1. Set up

Install [AC2](https://docs.appliedcompute.com/), enter `geo3k-vlm/`, and register
this project in your organization:

```bash
export AC2_ORG_ID='your-org-id'
ac2 project init
uv sync --locked
export AC2_PROJECT='geo3k-vlm'
```

## 2. Upload Geometry3K

```bash
uv run python upload_dataset.py
```

The uploader uses a pinned revision of
[hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) to create
`geo3k-train` (2,101 tasks) and `geo3k-validation` (300 tasks).
Questions and images become structured message content; expected answers go
in `grader_params`. The test split is left unused.

## 3. Train through Dispatch

The config in `src/geo3k_vlm/train.py` selects the
`Qwen/Qwen3.5-9B-VL` catalog model and `backend="dispatch"`. The `-VL` suffix
enables image training. AC2 supplies the training model to the agent and
resolves GPU topology from the catalog.

Push your code branch, then submit to your Dispatch cluster:

```bash
ac2 project push
uv run python -m geo3k_vlm.train --cluster YOUR_CLUSTER_ID
```

The example runs 20 training steps with one training replica and one inference
replica. Replicas can span multiple GPUs. It evaluates on the validation set
before training and every 10 steps.

## 4. View results

Use the printed training ID to inspect progress, image traces, and scores in
the AC2 platform, or follow logs with:

```bash
ac2 train get TRAIN_ID
ac2 train logs TRAIN_ID
```

See the [VLM guide](https://docs.appliedcompute.com/platform/training/vlm)
for more on image inputs.
