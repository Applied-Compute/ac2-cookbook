---
name: testing-ac2-cookbook
description: How to run and verify ac2-cookbook example evals locally against the prod AC2 platform from a Devin box, including auth env vars, venv layout, the tasks= path that avoids S3 task-blob reads, and remote dataset prerequisites.
---

# Testing ac2-cookbook evals end-to-end

Each example project (`tau2bench`, `dapo-math-check`, `rock-paper-scissors`, `harbor-acp`, `swebench-opencode`) is an independent uv project with its own `.venv` and `[tool.ac2]` marker in `pyproject.toml`. Prefer `<project>/.venv/bin/python -m <pkg>.eval` over `uv run` to avoid re-resolving the private index (which needs `UV_INDEX_AC2_USERNAME=__token__` + `UV_INDEX_AC2_PASSWORD`).

## Devin Secrets Needed

- `AC2_HOSTED_API_KEY` — personal AC2 API key; pass as `AC2_API_KEY` env to `ac2.sdk.Client` calls.
- `OPENAI_API_KEY` — org OpenAI key (verified working for `gpt-5-mini` via the Responses API).
- Optionally `UV_INDEX_AC2_*` if a `uv sync` is needed.

## Auth / endpoint gotcha

The box exports `AC2_MODE=local`, which makes `Client()` target `http://127.0.0.1:8000` (connection refused — no local control plane runs). Always prefix commands with `AC2_MODE=prod` (prod defaults to `https://api.appliedcompute.com`, no `~/.ac2/config` needed). Probe connectivity cheaply first:

```bash
curl -s -X POST https://api.appliedcompute.com/v1/auth/token -H "X-API-Key: $AC2_HOSTED_API_KEY"
```

## `--local` evals still hit the platform

`client.eval.run(config, local=True)` resolves the project, pushes a project version, registers a remote eval job (`local: true`), resolves dataset task refs, then runs rollouts in-process (model calls go straight to `api.openai.com` with `OPENAI_API_KEY`). Verify what the server stored via `client.eval.get(eval_id).launch_params["config"]["agent"]` — parameterized agents show up as `{"name": ..., "kwargs": {...}}`.

## S3 task blobs may be unreadable

`load_tasks` reads `s3://appliedcompute/ac2/tasks/...` directly with ambient AWS creds; the box's `devin-agent` IAM only allows `ac2/seed/askcode/v1/*`, so dataset-based `--local` evals fail with `s3:GetObject` AccessDenied after the eval job registers. Workaround that still exercises the whole component path: `EvalConfig(..., tasks=[Task(input=[Message(role="user", content=...)], env_params={"answer": ...}, grader_params={"answer": ...})], local=True)` — `tasks=` is local-only and skips the blob fetch.

## Remote prerequisites

Eval datasets must exist remotely under the project — list them with `GET /v1/datasets?project_id=<id>` (Bearer token from `/v1/auth/token`). As of 2026-09: `dapo-math-check-eval256` and `tau2bench-{airline,retail}-{train,test}` exist; `tau2bench-telecom-{train,test}` do NOT (only `-v1`-suffixed names) — `--domain telecom` fails dataset resolution until seeded.

## Cheap structural checks

`EvalConfig`/`TrainingConfig` are plain dataclasses — build configs, call `normalize_component_reference(cfg.agent)` for the captured `ComponentConstruction`, and `wire_orchestrator`/`resolve_orchestrator` to prove registry re-instantiation, all without network. Bad constructor values defer at `__init__` and raise `ComponentRegistryError` on submission. `DeploymentConfig.agent/env` are `str`-only (name-based by contract). Training must not be submitted for real — `train.run` starts cluster jobs.
