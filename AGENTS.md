# Agent Guide

This repository holds runnable AC2 examples. Most examples connect a self-hosted
harness to AC2: a Harness API server the user runs, the Harness Connector
(`ac2 harness connect`), and an eval and training job that call the harness.

## Where to look

- `README.md` lists the examples.
- `.claude/skills/ac2-companion/SKILL.md` describes the Harness API, the
  connector, and the eval and training workflow.
- `byoh-smoke-example/` is the smallest working harness. Read it before the
  larger examples.
- <https://docs.appliedcompute.com/llms.txt> indexes the AC2 docs.

## Working in this repository

- Each example is its own project. Run commands from the example's directory
  and follow its README in order.
- Check every command against `ac2 <command> --help` and every SDK call against
  the installed `ac2` package before running it.
- Do not create credentials, store secrets, or start eval or training jobs
  unless the user asks. Have the user enter secret values themselves.
- To build a new harness, copy `byoh-smoke-example/harness_server.py`, keep its
  six routes, and replace `run_rollout` and the scoring in `/submit_grading`.
