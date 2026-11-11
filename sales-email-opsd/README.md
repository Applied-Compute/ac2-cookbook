# Sales email style with OPSD

Teach a sales email assistant to avoid dash punctuation through teacher-only hints.
The student keeps the same MailPilot system prompt. This example isolates a measurable
style change; it does not establish that the emails are persuasive or factually correct.

`data/traces.jsonl` holds **150 synthetic conversations** generated with Claude
Sonnet 4.5, not customer production traces.

## Set up and upload the datasets

Install and authenticate [AC2](https://docs.appliedcompute.com/quickstart). Requires
Python 3.12+ and AC2 0.3.79+. From this directory:

```bash
ac2 projects init
uv sync
uv run python upload_dataset.py
```

The commands below default to `sales-email-opsd`; use `--project YOUR_PROJECT` on
upload, eval, train, and deploy commands for your own project.

| Dataset | Tasks | Use |
| --- | ---: | --- |
| `sales-email-opsd-train-v2` | 347 | Offline replay or one-step sampling |
| `sales-email-eval-v2` | 30 | Held-out conversations, with no hints |

`upload_dataset.py` builds every task from the conversations without model calls
(`build_tasks` in `dataset.py`). It validates alternating roles, unique conversation
IDs, and disjoint opening requests across the split, and selects up to three
offending replies per training conversation. Each replay task contains the original
system prompt, the
conversation prefix, and **only the selected assistant reply**. Its hint is a
`HintSelection` in `grader_params["opsd_hint_selections"]`: the prefix ends before
that reply, and the hint is appended to the rep's request; future turns are excluded.

Hints quote exact offending excerpts and ask the teacher to apply the shared style
policy while preserving meaning. A deterministic hint template is sufficient for
this mechanically detectable behavior. `SalesEmailJudgeHinter` in `hinter.py` has a
model write each hint instead: it quotes up to three offending phrases from the
selected reply and rewrites each one without the dash (see "Model-written hints").

Repeating an upload is a no-op, and an interrupted upload resumes missing tasks. If a
dataset already holds different tasks, the upload stops; choose new names in
`dataset.py` for a new experiment.

## Evaluate, train, and compare

Start with the same catalog model used for training:

```bash
uv run python -m sales_email_opsd.eval --model Qwen/Qwen3-4B
```

This is a remote eval that allocates a model server. For a small functional check
using a provider model, register that provider's key as an AC2 secret and run:

```bash
uv run python -m sales_email_opsd.eval --api-model gpt-5-mini --num-tasks 2
```

The provider-model check is not the Qwen training baseline.

Choose one training mode; `one_step` is the default:

```bash
uv run python -m sales_email_opsd.train_opsd --smoke
uv run python -m sales_email_opsd.train_opsd
uv run python -m sales_email_opsd.train_opsd --mode offline
```

| Mode | Student response | Teacher context |
| --- | --- | --- |
| `offline` | Recorded selected reply | Prefix plus stored hint |
| `one_step` | Newly sampled selected reply | Prefix plus stored hint |

Every mode allocates training compute. `--smoke` means one training step over at
most ten tasks, not an offline unit test. Defaults are Qwen3-4B, 20 steps, seed 42,
eight prompts per batch, and one sample per prompt. Training runs no eval; use
separate eval runs after training.

Find a checkpoint ID in the run's checkpoint list, then evaluate it:

```bash
ac2 -p sales-email-opsd train checkpoints TRAIN_ID
uv run python -m sales_email_opsd.eval --checkpoint CHECKPOINT_ID
```

Both remote eval commands use all 30 held-out tasks, one sample per task, and
temperature 0. The default user policy replays the same recorded requests and
followups for each model. This holds the user inputs fixed, although recorded
feedback can fit a new draft imperfectly. Use `--simulate-user` on eval for
adaptive conversations (`gpt-4.1` plays the rep); that adds provider cost and makes a
paired comparison less controlled. Simulator errors fail the rollout instead of
silently ending a successful conversation.

Deploy the agent on a provider model or a served endpoint:

```bash
uv run python -m sales_email_opsd.deploy --model gpt-5-mini
```

Compare the **score** (share of dash-free replies), `dashes_per_1k_words`, the strict
pass in `clean`, output validity, and run failures. Inspect drafts for preserved
facts, the intended call to action, and useful content. Thirty conversations give
a small demonstration, not a broad estimate of generalization.

## Model-written hints

To train on hints written by a model instead of the template, run the hinter over
the replay dataset. It calls `gpt-5-mini` once per task, so store the provider key
first: `ac2 secrets put --key OPENAI_API_KEY --value <val>`.

```bash
ac2 -p sales-email-opsd hinter run --hinter SalesEmailJudgeHinter --source-dataset sales-email-opsd-train-v2
uv run python -m sales_email_opsd.train_opsd --dataset SalesEmailJudgeHinter-hinted-sales-email-opsd-train-v2
```

The hinter run publishes `SalesEmailJudgeHinter-hinted-sales-email-opsd-train-v2`:
the same replay tasks with each stored hint replaced by the model's. The source
dataset is unchanged.

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
