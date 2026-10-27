# Sales email style with OPSD

Teach a sales email assistant to avoid dash punctuation through teacher-only hints.
The student keeps the same MailPilot system prompt. This example isolates a measurable
style change; it does not establish that the emails are persuasive or factually correct.

The bundled corpus contains **150 synthetic conversations**, not customer production
traces. You can rebuild all datasets without credentials or model calls.

## Set up and check locally

Install and authenticate [AC2](https://docs.appliedcompute.com/quickstart), then run
from this directory:

```bash
uv sync
uv run python build_opsd_dataset.py
uv run python upload_dataset.py --check
```

Requires Python 3.12+ and AC2 0.3.73+.

For a new AC2 project, register this folder with `ac2 projects init` before uploading.
The commands below default to `sales-email-opsd`; use `--project YOUR_PROJECT` on
upload, eval, train, and deploy commands for your own project.

## Upload the datasets

```bash
uv run python upload_dataset.py
```

| Dataset | Tasks | Use |
| --- | ---: | --- |
| `sales-email-opsd-train-v2` | 347 | Offline replay or one-step sampling |
| `sales-email-opsd-online-train-v2` | 120 | Full conversations with live hints |
| `sales-email-eval-v2` | 30 | Held-out conversations, with no hints |

The builder validates alternating roles, unique conversation IDs, and disjoint
opening requests across the split. It selects up to three offending replies per
training conversation. Each replay task contains the original system prompt, the
conversation prefix, and **only the selected assistant reply**. Its hint is a
`HintSelection` in `grader_params["opsd_hint_selections"]`: the prefix ends before
that reply, and the hint is appended to the rep's request; future turns are excluded.

Hints quote exact offending excerpts and ask the teacher to apply the shared style
policy while preserving meaning. A deterministic hint template is sufficient for
this mechanically detectable behavior. Online mode uses a model to write the hint
from the live rollout instead: `SalesEmailJudgeHinter` in `hinter.py`, passed to
training as `rmsd_hinter_path`, picks the first reply that uses dash punctuation,
quotes up to three offending phrases, and rewrites each one without the dash.

`data/manifest.json` records source and output hashes. Upload validates the files
against both the manifest and the current builder. Repeating an upload is a no-op;
an interrupted upload resumes missing tasks. A dataset containing a different
snapshot causes an error. Choose new names in `dataset.py` for a new experiment.

## Evaluate, train, and compare

Start with the same catalog model used for training:

```bash
uv run python -m sales_email_opsd.eval --model Qwen/Qwen3-4B
```

This is a remote eval that allocates a model server. For a small functional check
using a provider model, register that provider's key and run:

```bash
uv run python -m sales_email_opsd.eval --local --api-model gpt-5-mini --num-tasks 2
```

The provider-model check is not the Qwen training baseline. Local eval still reads
the registered AC2 dataset and emits AC2 traces.

Choose one training mode:

```bash
uv run python -m sales_email_opsd.train_opsd --mode offline --smoke
uv run python -m sales_email_opsd.train_opsd --mode one_step
uv run python -m sales_email_opsd.train_opsd --mode online
```

| Mode | Student response | Teacher context |
| --- | --- | --- |
| `offline` | Recorded selected reply | Prefix plus stored hint |
| `one_step` | Newly sampled selected reply | Prefix plus stored hint |
| `online` | Newly sampled full conversation | Live prefix plus generated hint |

Every mode allocates training compute. `--smoke` means one training step over at
most ten tasks, not an offline unit test. Defaults are Qwen3-4B, 20 steps, seed 42,
eight prompts per batch, and one sample per prompt. Online mode evaluates the held-out
set before training and during it. Offline and one-step mode use separate eval runs
after training.

Online mode simulates the rep with `gpt-4.1` so each conversation reacts to the
student's drafts; `--replay-user` replays the recorded rep turns instead. It also
uses `gpt-5-mini` twice: as the hint writer, and as a token judge that selects which
response tokens receive RMSD loss (`--no-token-judge` trains on every token). Store
`OPENAI_API_KEY` in your AC2 project's secrets.

Find a checkpoint ID in the run's checkpoint list, then evaluate it:

```bash
ac2 -p sales-email-opsd train checkpoints TRAIN_ID
uv run python -m sales_email_opsd.eval --checkpoint CHECKPOINT_ID
```

Both remote eval commands use all 30 held-out tasks, one sample per task, and
temperature 0. The default user policy replays the same recorded requests and
followups for each model. This holds the user inputs fixed, although recorded
feedback can fit a new draft imperfectly. Use `--simulate-user` on eval for
adaptive conversations like online training's; that adds provider cost and makes a
paired comparison less controlled. Simulator errors fail the rollout instead of
silently ending a successful conversation.

Compare the **score** (share of dash-free replies), `dashes_per_1k_words`, the strict
pass in `clean`, output validity, and run failures. Inspect drafts for preserved
facts, the intended call to action, and useful content. Thirty conversations give
a small demonstration, not a broad estimate of generalization.

With the defaults (online, 20 steps), the held-out score went from 0.10 before
training to 0.32 at step 10 and 0.20 at step 20; dashes per conversation fell from
6.1 to 3.5 at step 10. A second run reproduced the peak (0.09 to 0.29 at step 10).
The gain peaks around step 10 and partly fades by step 20.

## What the grader measures

The shared policy flags em dashes, double hyphens and en dashes between prose,
and spaced hyphens used as sentence breaks. It preserves hyphenated compounds,
hyphen/en-dash numeric ranges, list markers, and Markdown rules.

The score is the share of the model's replies that are nonempty and use no dash
punctuation, so each reply gets partial credit and a long reply cannot dilute a dash.
Only the visible reply is graded: reasoning items, a leading `<think>` block (Qwen3
returns its reasoning inline), and user messages are excluded. Email shape, truncation,
and the strict all-replies pass (`clean`) are recorded as diagnostics alongside the
score; the subject/body guard is deliberately minimal and is not a semantic quality grader.

In OPSD, this score provides telemetry and evaluation; the learning signal comes
from the hinted teacher distribution, not from optimizing this score.

## Generate a new corpus (optional)

Bundled data is ready to use. To generate a separate experiment through a provider
API, set its key in your environment or `.env` and run:

```bash
uv run python generate_traces.py --data-dir data/new-experiment --model gpt-4.1-2025-04-14
uv run python build_opsd_dataset.py --data-dir data/new-experiment
uv run python upload_dataset.py --data-dir data/new-experiment --check
```

Generation incurs model usage. Train/eval persona seeds are independent, so growing
the training split does not move the eval split. IDs include the model, prompts,
seed, and persona; resuming with incompatible settings fails. Completed conversations
are checkpointed. A fixed seed fixes personas, not stochastic model responses;
keep the generated source file to reproduce the dataset exactly.

## Files

Runtime components and config factories live in `src/sales_email_opsd/`. The three
root scripts generate, build, and upload data. The bundled conversations were
generated with Claude Sonnet 4.5.
