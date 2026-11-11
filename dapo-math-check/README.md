# dapo-math-check

Math-answer checking on AC2: the agent solves a math problem with two tools —
`check_answer` (verify a candidate, returning "Correct."/"Incorrect.", at most
three incorrect submissions) and `finish` (end the task with a final answer,
graded directly).

## Setup

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow set up [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

```bash
ac2 project init
uv sync
```

## Register datasets

```bash
uv run python upload_dataset.py
```

## Run an eval

Register your provider key as a secret first:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <val>
uv run python -m dapo_math_check.eval
```

## Submit a training run

```bash
uv run python -m dapo_math_check.train
```

## Deploy

```bash
uv run python -m dapo_math_check.deploy
```

## Layout

```text
dapo-math-check/
├── pyproject.toml              # package metadata + [tool.ac2]
├── upload_dataset.py           # register the base train/eval datasets
├── data/                       # bundled train/eval parquet files
└── src/dapo_math_check/
    ├── agent.py
    ├── environment.py          # check_answer + finish tools
    ├── user.py
    ├── grader.py
    ├── answer_utils.py
    ├── eval.py                 # named EvalConfig + launch
    ├── train.py                # named TrainingConfig + launch (GRPO)
    ├── train_opsd.py           # offline / one_step / online OPSD configs
    ├── train_sft.py            # SFT config on the OPSD transcripts
    └── deploy.py               # named DeploymentConfig + launch
```

Pass `--model <name>` to `dapo_math_check.eval` to eval a different model.
Edit `eval.py` / `train.py` for sampling, dataset caps
(`DatasetSource(..., num_tasks=...)`), or replica counts.

## OPSD (self-distillation)

On-policy self-distillation on this same task. The failure it targets is
**premature finalization**: the base model ends the task — via `finish`, or a
plain-text answer — before either getting a "Correct." from `check_answer` or
spending all three attempts. A teacher-only hint tells it to keep verifying with
`check_answer` first, and RMSD distills the corrected behavior against the frozen
base as teacher. The behavior is taught only through the hint, never the student
prompt.

These configs need two datasets that `upload_dataset.py` does not create, and
their source transcripts are not bundled in `data/`. Register your own under
these names before running the commands below:

- `dapo-math-check-opsd-train` — base-model transcripts of premature
  finalizations. Each task's `input` is the transcript, and
  `grader_params["opsd_hint_selections"]` holds the `HintSelection`: the
  teacher-only hint and the assistant-turn boundary to distill. Used by offline,
  one-step, and SFT.
- `dapo-math-check-opsd-online-train` — the plain questions those failures came
  from. Used by online, which also needs an `OnlineHinter` to write the hint
  live at training time; this example does not ship one.

[`sales-email-opsd`](../sales-email-opsd) is a complete OPSD example: it builds
its hinted datasets and ships an `OnlineHinter`.

### Offline

Replays the stored boundary step: the teacher sees prompt + hint, the student
replays its recorded response. No inference engines.

```bash
uv run python -m dapo_math_check.train_opsd --mode offline
```

### One-step

Same dataset, but the student samples the boundary step live from the current
policy instead of replaying it (the teacher still sees prompt + hint). Inference
engines run.

```bash
uv run python -m dapo_math_check.train_opsd --mode one_step
```

### Online

The student rolls out the full episode live — the real `DapoMathAgent` /
`DapoMathCheckEnvironment` / `DapoMathCheckGrader` — and an `OnlineHinter`
inspects each graded rollout, picks the assistant turn that finalized
prematurely, and writes the teacher-only hint for it.

```bash
uv run python -m dapo_math_check.train_opsd --mode online
```

## SFT

Supervised fine-tuning on the same stored transcripts
(`dapo-math-check-opsd-train`):

```bash
uv run python -m dapo_math_check.train_sft
```

## RMSD token-selection judge (orthogonal to rollout mode)

All three OPSD modes optimize the same RMSD loss (top-K reverse-KL, student vs
frozen teacher over the shared response). The token-selection judge is a
separate, optional knob narrowing which response tokens that loss trains on — set
`judge_model` (and the `judge_*` fields) on any config in `train_opsd.py`. It
composes with, but is distinct from, the online hinter. The `online`
config ships it on (`judge_model="gpt-5-mini"`); delete that line for the plain
RMSD loss.
