# Lichess puzzles

A deterministic multi-turn chess environment built from the public
[Lichess puzzle database](https://database.lichess.org/#puzzles). The agent sees a
position and has one `submit_move` tool. Correct moves advance through the recorded
line; incorrect moves end the puzzle. The grader gives normalized prefix progress and
a completion bonus.

## Setup

Register a project that you own before uploading a dataset or submitting a run. This
example intentionally never selects the shared cookbook project for you.

```bash
ac2 project init
uv sync
```

Use that project name in every remote command below as `<your-project>`.

## Build and register the dataset

The source database is CC0 and is not committed. Download it once, convert it into
reproducible train/eval task files, then register those immutable task files in your
own AC2 project.

```bash
uv run python -m construction.download_lichess
uv run python -m construction.prepare_datasets \
  --output-dir data/prepared/final-v1 \
  --rating-bins 800-1199,1200-1599 \
  --length-bins 2,3,4,5-6 \
  --split eval:1000:2000:100 \
  --split train:2000:10000:1000
uv run python -m construction.register_datasets \
  --project <your-project> \
  --manifest data/prepared/final-v1/manifest.json \
  --split eval --split train
```

This creates 800 held-out evaluation puzzles and 8,000 training puzzles, balanced across
rating and solution-length strata. The prepared manifest records source provenance, split
definitions, and exact task counts. Rerun registration with `--allow-existing` only to
resume an interrupted upload of the same prepared dataset.

## Remote evaluation

First print the resolved request without contacting AC2:

```bash
uv run python -m lichess_puzzles.eval --project <your-project>
```

Add `--submit` to run it. Select `--variant qwen36` to use the self-served Qwen
policy, or pass `--num-tasks N` for a bounded dataset slice.

## GRPO training

The launcher uses 16 prompts per rollout batch and 8 samples per prompt. It enables
periodic evaluation sidecars and defaults global sampling concurrency to 128.

```bash
uv run python launch/train_grpo.py --project <your-project> --dry-run
```

Omit `--dry-run` to submit the training run. Use `--cluster`, `--gpu-type`, or
`--image` when your project needs a particular execution configuration.

## Evaluate a saved checkpoint

Checkpoint evaluation is also dry-run by default. Fill in locations produced by your
training run, inspect the request, and add `--submit` to schedule it.

```bash
uv run python launch/eval_checkpoint.py \
  --project <your-project> \
  --train-id train_xxx \
  --checkpoint-dir s3://... \
  --hf-checkpoint s3://... \
  --iter-num 100
```

## Layout

```text
lichess-puzzles/
├── construction/                 # public-data download, preparation, and registration
├── launch/                       # Dispatch GRPO and checkpoint-eval entry points
└── src/lichess_puzzles/          # agent, environment, orchestrator, grader, and configs
```
