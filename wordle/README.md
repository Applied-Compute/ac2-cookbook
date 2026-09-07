# Wordle

Train **Qwen3-4B** to find a hidden five-letter word using `check_answer(guess)`.
Each task supplies one `base_word` through environment parameters; the model
sees the same prompt for every task.

The tool trims surrounding whitespace, accepts ASCII letters in either case,
and checks the bundled [ENABLE dictionary](src/wordle/data/README.md). It returns
`invalid guess` or five tiles: 🟩 correct position, 🟨 misplaced letter, ⬜ absent
unmatched copy. Greens consume answer letters first; yellows consume remaining
copies from left to right ([duplicate-letter examples](https://www.classes.cs.uchicago.edu/archive/2023/fall/22300-1/hello-again-wordle/)).

A correct guess or six attempts ends the rollout. Invalid submissions count as
attempts. Wins on guesses 1–6 earn **1.0, 0.9, 0.8, 0.7, 0.6, 0.5**; losses,
invalid guesses, and unfinished games earn zero.
Premature text answers receive up to three reminders to use the tool. Reminders
do not count as guesses or reset the rollout's token budget; continued refusal
ends as an unfinished game with zero reward.

## Set up and test

Install and authenticate [AC2](https://docs.appliedcompute.com/quickstart), then
run from this folder:

```bash
export AC2_MODE=prod
uv sync
uv run python -m unittest discover -s tests -v
ac2 project init
uv run python upload_dataset.py
```

The upload creates `wordle-train2048` and `wordle-eval256` in project `wordle`.
A fixed seed of 42 gives disjoint, reproducible answer sets. The full dictionary
remains available for guess validation. Uploading again deduplicates tasks.

## Evaluate and train

For a local smoke test, supply a configured inference model and its provider
credentials. For an OpenAI-compatible service, also pass `--base-url API_URL`
and `--api-key-env KEY_VARIABLE`. Remote eval starts an ephemeral Qwen3-4B engine
on the default dispatch cluster, which lives until the eval ends. Training uses
Modal; AC2's `EvalServe` currently requires the dispatch backend.

```bash
uv run python -m wordle.eval --local --model INFERENCE_MODEL --num-tasks 8
uv run python -m wordle.eval
uv run python -m wordle.train --dry-run
uv run python -m wordle.train
```

For **Qwen3.6-35B-A3B** on the same datasets and training schedule:

```bash
uv run python -m wordle.train_qwen36 --dry-run
uv run python -m wordle.train_qwen36
```

This uses one eight-B200 node: a four-GPU training replica (TP4, EP4, CP1)
and two two-GPU inference replicas. Optimizer state is offloaded to CPU.
Each checkpoint eval uses two additional B200s. The Qwen3.6 agent uses its
native XML function-call format with thinking enabled. The launcher verifies
the submitted allocation and stops the run if it differs. The same 55-minute
deadline covers submission, startup, training, and attached evaluations.
Keep the launcher running; its PID and absolute deadline are saved with the run.

Training uses Modal B200s, four training and four inference replicas, 40 GRPO
steps, eight words per batch, and four samples per word. Context parallelism is
explicitly set to one to keep the training allocation at eight GPUs. A baseline
eval and evals after every five steps each use one additional B200 and the same
256 held-out words, with one sample per word. All eight training checkpoints
are retained so queued evals can read them. Qwen3 thinking is enabled in both
eval and training.
Training and checkpoint evals allow 24,576 response tokens across the whole
game, a total context of 32,768 tokens, and up to 600 seconds per rollout.
Standalone eval allows 8,192 output tokens per turn.
AC2 builds and uploads the project before submitting;
commit and push your branch first to make the source easy to reproduce. Qwen3-4B
weights must be available to the Modal backend. Training uses AC2 credentials
and requires no external model-provider key.

The remote smoke eval uses 32 held-out words. Check game completion as well as
reward: zero reward can mean either a completed loss or an unfinished rollout.
After exporting traces with `client.traces.download(EVAL_ID, dest="runs/eval.jsonl")`,
run `uv run python -m wordle.audit runs/eval.jsonl`. It exits nonzero for token
limit errors, unfinished games, early abandonment, or missing grades.

**Keep the training launcher running.** It monitors training and attached evals
and requests a stop after 55 minutes including submission/startup, leaving five
minutes for shutdown. The deadline can stop this run before all 40 steps finish.
It also stops the run on Ctrl-C or a polling error. This is a client-side deadline;
closing or killing the launcher removes that protection. Use `--max-minutes 20`
for a shorter budget. Run IDs and submitted configs are saved under `runs/`.

Inspect the printed training ID with these commands:

```bash
ac2 train get TRAIN_ID
ac2 train logs TRAIN_ID
ac2 train checkpoints TRAIN_ID
ac2 train stop TRAIN_ID
```

Modal saves the final checkpoint under `/data/ac2/wordle/TRAIN_ID/iter_0000039`
on its persistent data volume. The checkpoint-list API may return an empty list
for Modal runs; verify the volume files before relying on that list.
