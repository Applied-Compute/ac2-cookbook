# AC2 Cookbook

Runnable examples for evaluating and training agents on
[AC2](https://docs.appliedcompute.com/) with your own harness.

Each example in the table below is a
[self-hosted harness](https://docs.appliedcompute.com/platform/custom-harnesses/self-hosted):
the agent and its tools run on your infrastructure, while AC2 serves the model
and runs the eval or training job.

## Getting started

Install AC2 ([docs](https://docs.appliedcompute.com)): have an agent follow
[agents.md](https://platform.appliedcompute.com/agents.md), or run
`curl -fsSL https://api.appliedcompute.com/install.sh | sh`. The installer logs
you in and gives `uv` access to the AC2 package index, which every example's
`uv sync` installs `ac2` from. If `uv sync` cannot authenticate, run
`ac2 login` again.

You also need:

- Harness Connector credentials. An organization admin creates them once with
  `ac2 harness credentials create`.
- An OpenAI API key. The baseline evals call `gpt-5.6-luna`.
- A [Modal](https://modal.com) account for `swebench-opencode` and
  `terminal-bench-harbor`, which run their sandboxes there.

Start with [byoh-smoke-example](byoh-smoke-example/). It checks your
credentials, network, and connector with three questions before you run a
benchmark.

## Examples

Apart from the smoke example, each one uses a public benchmark. They follow the
same steps: upload the dataset, run a baseline eval, then run a short training
job. The Terminal-Bench example stops at the eval.

| Directory | Benchmark | Shows |
| --- | --- | --- |
| [byoh-smoke-example](byoh-smoke-example/) | Three factual questions | The smallest harness: one model call per rollout |
| [swebench-opencode](swebench-opencode/) | SWE-bench Verified | A coding agent (OpenCode) |
| [tau2bench](tau2bench/) | tau2-bench (airline, retail) | A customer-service agent |
| [terminal-bench-harbor](terminal-bench-harbor/) | Terminal-Bench | A terminal agent run through Harbor on Modal (eval only) |
| [geo3k-vlm](geo3k-vlm/) | Geometry3K | Training a vision-language model |

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

The [custom harness overview](https://docs.appliedcompute.com/platform/custom-harnesses)
describes the Harness API and the connector. For the grading endpoints, see
[`byoh-smoke-example/harness_server.py`](byoh-smoke-example/harness_server.py).

## Other examples

These do not use a self-hosted harness.

- [dapo-math-check](dapo-math-check/): math-answer checking with tools, eval, and training.
- [sales-email-opsd](sales-email-opsd/): teach a sales email agent a style rule with OPSD hints.
- [data-pipelines](data-pipelines/): the patterns in the [AC2 pipelines guide](https://docs.appliedcompute.com/platform/data/pipelines).
