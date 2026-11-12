# AC2 Cookbook

Runnable examples for evaluating and training agents on
[AC2](https://docs.appliedcompute.com/) with your own harness.

Every example here is a
[self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted):
the agent, its tools, and the grading run on your infrastructure, while AC2
serves the model and runs the eval or training job.

## How a self-hosted harness connects to AC2

- **Harness API**: an HTTP server you run. AC2 calls it to start a rollout (one
  attempt at one task), check its status, and collect its outputs.
- **Harness Connector**: `ac2 harness connect`, run next to your server. It makes
  an outbound connection to AC2 and forwards requests to your Harness API, so the
  server stays private.
- **Model endpoint**: each rollout request carries a URL and key. Your agent
  sends its model calls there; AC2 serves the model selected for the job (in
  training, the model being trained) and records the trace.
- **Grading**: your harness scores each rollout and returns the score to AC2.

Read the [custom harness overview](https://docs.appliedcompute.com/platform/custom-harnesses)
for the full contract.

## Examples

Each example uses a public benchmark and follows the same steps: upload the
dataset, run a baseline eval, run a short training job, then eval the
checkpoint.

| Directory | Benchmark | Shows |
| --- | --- | --- |
| [byoh-smoke-example](byoh-smoke-example/) | Three factual questions | A connectivity check: one model call per rollout, shipped as a container. Start here. |
| [swebench-opencode](swebench-opencode/) | SWE-bench Verified | A coding agent (OpenCode) |
| [tau2bench](tau2bench/) | tau2-bench (airline, retail, telecom) | A customer-service agent |
| [terminal-bench-harbor](terminal-bench-harbor/) | Terminal-Bench | A terminal agent run through Harbor on Modal |
| [hotpotqa-search-agent](hotpotqa-search-agent/) | HotpotQA | A search agent built on the OpenAI Agents SDK |
| [geo3k-vlm](geo3k-vlm/) | Geometry3K | Training a vision-language model |

## Getting started

Install AC2 first ([docs](https://docs.appliedcompute.com)): have an agent follow
[agents.md](https://platform.appliedcompute.com/agents.md), or run
`curl -fsSL https://api.appliedcompute.com/install.sh | sh`.

The Harness Connector needs its own credentials. An organization admin creates
them once:

```bash
ac2 harness credentials create
```

Then enter an example, register it with `ac2 project init`, and follow its
README.

## Other examples

These do not use a self-hosted harness.

- [dapo-math-check](dapo-math-check/): math-answer checking with tools, eval, and training.
- [sales-email-opsd](sales-email-opsd/): teach a sales email agent a style rule with OPSD hints.
- [data-pipelines](data-pipelines/): the patterns in the [AC2 pipelines guide](https://docs.appliedcompute.com/platform/pipelines).

## Links

- [AC2 docs](https://docs.appliedcompute.com/)
- [AC2 platform](https://platform.appliedcompute.com/)
