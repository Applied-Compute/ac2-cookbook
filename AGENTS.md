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

## Editing guidelines

- Keep each example's README short and runnable: upload the dataset, run a
  baseline eval, run a short training job.
- Keep cluster names, IDs, and secrets out of READMEs.
- Check every command against `ac2 <command> --help` and every SDK call against
  the installed `ac2` package before writing it down.
- State the project, dataset, model, and secrets each command needs.
- After editing an example, run its narrowest relevant check and report what
  was run.
