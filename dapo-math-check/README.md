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

This registers every dataset the commands below use:

| Dataset | Tasks | Source | Used by |
| --- | ---: | --- | --- |
| `dapo-math-check-train2048` | 2048 | `data/dapo-mid-train2048.parquet` | GRPO training |
| `dapo-math-check-eval256` | 256 | `data/dapo-mid-eval256.parquet` | Eval, GRPO eval |
| `dapo-math-check-opsd-train` | 90 | `data/opsd-transcripts.jsonl` | SFT, OPSD |

## Run an eval

Register your provider key as a secret first:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <val>
uv run python -m dapo_math_check.eval
```

The eval runs `gpt-5.6-luna` on 10 tasks. Pass `--model <name>` to eval a
different model.

## Train with GRPO

```bash
uv run python -m dapo_math_check.train
```

## Train with SFT

Supervised fine-tuning on the stored transcripts in
`dapo-math-check-opsd-train`: each task's `input` is a full conversation, and
the loss covers its assistant turns.

```bash
uv run python -m dapo_math_check.train_sft
```

The transcripts are base-model rollouts, so this run shows the SFT mechanics
rather than a better policy. Swap in transcripts you want the model to imitate.

## Train with OPSD (self-distillation)

On-policy self-distillation on this same task. The failure it targets is
**premature finalization**: the base model ends the task — via `finish`, or a
plain-text answer — before either getting a "Correct." from `check_answer` or
spending all three attempts. A teacher-only hint tells it to keep verifying with
`check_answer` first, and RMSD distills the corrected behavior against the frozen
base as teacher. The behavior is taught only through the hint, never the student
prompt.

Each task in `dapo-math-check-opsd-train` is a base-model transcript that ends
with a premature finalization. `env_params["tools"]` holds the tool schemas, and
`grader_params["opsd_hint_selections"]` holds one `HintSelection`: the turn to
distill (`rollout_prefix_len`) and the hint, which the teacher sees appended to
the message just before that turn.

### Offline

Replays the stored turn: the teacher sees prompt + hint, the student replays its
recorded response. No inference engines.

```bash
uv run python -m dapo_math_check.train_opsd --mode offline
```

### One-step

Same dataset, but the student samples the turn live from the current policy
instead of replaying it (the teacher still sees prompt + hint). Inference
engines run.

```bash
uv run python -m dapo_math_check.train_opsd --mode one_step
```

## Deploy

```bash
uv run python -m dapo_math_check.deploy
```

## Layout

```text
dapo-math-check/
├── pyproject.toml              # package metadata + [tool.ac2]
├── upload_dataset.py           # register the train, eval, and transcript datasets
├── data/                       # bundled parquet problems + OPSD/SFT transcripts
└── src/dapo_math_check/
    ├── agent.py
    ├── environment.py          # check_answer + finish tools
    ├── user.py
    ├── grader.py
    ├── answer_utils.py
    ├── eval.py                 # named EvalConfig + launch
    ├── train.py                # named TrainingConfig + launch (GRPO)
    ├── train_sft.py            # SFT config on the stored transcripts
    ├── train_opsd.py           # offline / one_step OPSD configs
    └── deploy.py               # named DeploymentConfig + launch
```

Edit `eval.py` / `train.py` for sampling, dataset caps
(`DatasetSource(..., num_tasks=...)`), or replica counts.
