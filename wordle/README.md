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
uv run python -m wordle.eval --local --model INFERENCE_MODEL
uv run python -m wordle.eval
uv run python -m wordle.train --dry-run
uv run python -m wordle.train
```

Training uses Modal B200s, four training and four inference replicas, 10 GRPO
steps, eight words per batch, and four samples per word. Training-time eval is
off for this small run. AC2 builds and uploads the project before submitting;
commit and push your branch first to make the source easy to reproduce. Qwen3-4B
weights must be available to the Modal backend. Training uses AC2 credentials
and requires no external model-provider key.

**Keep the training launcher running.** It monitors the job and requests a stop
after 55 minutes including submission/startup, leaving five minutes for shutdown.
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
