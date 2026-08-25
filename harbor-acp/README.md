# Harbor ACP with AC2-managed lifecycle

This project runs Harbor's public
[`harbor/hello-world@latest`](https://hub.harborframework.com/datasets/harbor/hello-world/latest)
dataset with OpenCode through Harbor's
[generic ACP agent runner](https://www.harborframework.com/docs/agents/acp).
Harbor owns dataset resolution, the task sandbox, agent loop, and verifier; AC2
provides an OpenAI-compatible model relay. OpenCode sends model requests to
this endpoint; AC2 routes them to the model selected for the job and records
the trace. AC2 uses Harbor's verifier reward for eval or training. The
project-defined orchestrator runs inside the AC2 job and owns the rollout
lifecycle.

OpenCode is configured with AC2 as an OpenAI-compatible provider, so the same
harness works with an API model in eval and an AC2-hosted model in training.

## What this example teaches

This is the
[AC2-managed lifecycle](https://docs.appliedcompute.com/platform/custom-harnesses/ac2-managed)
for **adapting an existing harness framework**:

- Your thin orchestrator runs inside the AC2 job.
- Harbor owns dataset resolution, sandbox creation, agent execution, and
  verification.
- The orchestrator stores Harbor's `TrialResult` on its environment.
- The AC2 grader passes through Harbor's verifier reward.

Unlike the SWE-bench example, the orchestrator does not manually construct a
grading sandbox. It demonstrates how little adapter code is needed when an
existing harness already owns the full task lifecycle.

There is no Harness API server or Harness Connector in this deployment model.

## What you implement

[`src/harbor_acp/orchestrator.py`](src/harbor_acp/orchestrator.py) is the code a
you implement. `HarborACPOrchestrator` maps the lifecycle
operations to Harbor:

- `trigger_rollout` creates a Harbor job with the relay credentials supplied by
  AC2. The orchestrator generates and returns the rollout ID; AC2 does not
  assign it.
- `check_status` translates the Harbor task state into the BYOH status contract.
- `fetch_assets` stores the completed `TrialResult` for the grader.

## How a rollout moves through the example

1. AC2 loads the Harbor dataset reference from the task's `env_params`.
2. The job calls `trigger_rollout`, which starts one Harbor task on Modal.
3. OpenCode sends model requests through the AC2 relay while Harbor manages ACP
   messages, shell tools, and the task container.
4. Harbor runs its bundled verifier.
5. `fetch_assets` stores the result, and `HarborRewardGrader` returns the
   verifier's numeric `reward` to AC2.

## Code map

| File | Responsibility |
| --- | --- |
| `src/harbor_acp/orchestrator.py` | Your adapter around Harbor |
| `src/harbor_acp/environment.py` | Dataset configuration and completed Harbor result |
| `src/harbor_acp/grader.py` | AC2-side pass-through of Harbor's verifier reward |
| `upload_dataset.py` | AC2 dataset entry that identifies the Harbor Hub task |
| `src/harbor_acp/eval.py` | Eval configuration |
| `src/harbor_acp/train.py` | Bounded GRPO training configuration |

## Adapt it to your harness

Replace the Harbor job construction with your harness's Python or CLI entry
point. Preserve the three orchestrator methods, pass the relay credentials to
the model client unchanged, and store the final artifacts or verifier result on
the environment for your grader.

## Setup

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow [agents.md](https://platform.appliedcompute.com/agents.md), or run `curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

From this directory:

```bash
ac2 project init
uv sync
uv run modal setup
```

Add your Modal credentials to the project:

```bash
ac2 secrets put --key MODAL_TOKEN_ID --value <token-id>
ac2 secrets put --key MODAL_TOKEN_SECRET --value <token-secret>
```

BYOH is currently in beta. Contact the AC team to enable it for this project.
We will provision the required model relay and project configuration.

For evals against an OpenAI model, store the provider key in the project:

```bash
ac2 secrets put --key OPENAI_API_KEY --value <openai-api-key>
```

## Upload the task

The AC2 dataset entry points the orchestrator at Harbor's public Hub dataset;
Harbor resolves the dataset, downloads its task, and runs its bundled verifier:

```bash
uv run python upload_dataset.py
```

Commit and push the project before launching a remote job so AC2 can load its
components. The eval and training launchers use the model access provisioned by
AC2 automatically.

## Run an eval

```bash
uv run python -m harbor_acp.eval
```

## Run a bounded training smoke test

```bash
uv run python -m harbor_acp.train
```

The training command runs one Qwen3-4B step with two samples of the single
hello-world problem.
