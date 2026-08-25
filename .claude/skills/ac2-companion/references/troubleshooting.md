# Troubleshooting

Symptom → likely cause. Check here before inventing a new diagnosis.

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| 404 / "project not found" | Folder not initialized, or `Client(project=...)` mismatches `[project].name` | Run `ac2 project init` inside the project folder; keep `Client(project=...)` aligned with `[project].name`. |
| `uv sync` auth error on `ac2` | Package index credential missing/expired | Rerun `ac2 login`, then re-export `UV_INDEX_AC2_USERNAME` / `UV_INDEX_AC2_PASSWORD` if needed. |
| Remote run fails with provider auth | Secrets not registered | `ac2 secrets put --key OPENAI_API_KEY --value ...` |
| `ModuleNotFoundError` in remote run | Dependency missing from `[project].dependencies`, or package layout wrong | Add the dep to `pyproject.toml`, keep code under `src/<package>/`, then re-run. |
| Remote run doesn't reflect recent edits | Stale uploaded version | Re-run the project module (it pushes), or run `ac2 project push`. |
| Remote eval rejects `tasks=` | Inline tasks are local-only | Run the eval module with `--local`, or upload a dataset and set `dataset=` in `EvalConfig`. |
| Empty local eval results | Dataset empty or every rollout errored | Check dashboard traces; verify `upload_dataset.py` ran. |
| Deployment stuck `deploying` | Wheel/deps failed to install | Check deployment logs; fix `[project].dependencies` and re-run the deploy module. |
| Component name not found | Class is outside `src/<package>`, abstract, or has a different class name | Keep it in the managed package and use its exact class name in the config. |
| Duplicate component name | Two discovered classes of one component kind share a class name | Rename one class; names must be unique within a project. |
| Component constructor rejected | A component constructor requires arguments | Define `__init__(self)` with no arguments and move varying data to class variables or task params. |

## Debug order

1. Confirm the project marker: `[tool.ac2]` in `pyproject.toml`, and `ac2 project init` already run.
2. Confirm the config module loads: `uv run python -c 'from package.eval import CONFIG; print(CONFIG)'`.
3. Re-run the eval module locally first. If local works and remote fails, the delta is almost always secrets or packaging deps.
4. Check dashboard traces for the run ID printed by the CLI.
