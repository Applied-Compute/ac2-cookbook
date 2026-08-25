# Agent Guide

Use this repository as a cookbook for AC2 starter projects. Keep user-facing
READMEs short and runnable; put agent workflow guidance here.

## Reference Path

1. Read `.claude/skills/ac2-companion/SKILL.md` for the AC2 workflow.
2. Use `.claude/skills/ac2-companion/references/setup.md` for install and CLI setup.
3. Use `.claude/skills/ac2-companion/references/runtime.md` for the `Agent`,
   `Environment`, `Task`, `Grader`, and `Orchestrator` model.
4. Use `.claude/skills/ac2-companion/references/datasets.md` before eval or training work.
5. Use `.claude/skills/ac2-companion/references/eval-local.md` and
   `.claude/skills/ac2-companion/references/eval-remote.md` for eval workflows.
6. Use `.claude/skills/ac2-companion/references/training.md` for training submissions.
7. Use `.claude/skills/ac2-companion/references/migrations.md` before porting another
   framework into AC2.

## Project Order

1. `rock-paper-scissors/` for local sessions, tools, grading, and deployment.
2. `dapo-math-check/` for datasets, custom user policy, eval, and training.
3. `tau2bench/` for simulated users, domain environments, remote eval, and training.

## Editing Guidelines

- Prefer existing AC2 primitives and local project patterns over new scaffolding.
- Keep remote eval and training instructions explicit about projects, datasets, secrets,
  commits, and pushed branches.
- Avoid internal-only launchers, monorepo paths, or infrastructure names in user-facing
  READMEs.
- After editing a project, run the narrowest relevant tests or syntax checks for that
  project and report what was run.
# Agent Guide

Use this repository as a cookbook for AC2 starter projects. Keep user-facing
READMEs short and runnable; put agent workflow guidance here.

## Reference Path

1. Read `.claude/skills/ac2-companion/SKILL.md` for the AC2 workflow.
2. Use `.claude/skills/ac2-companion/references/setup.md` for install and CLI setup.
3. Use `.claude/skills/ac2-companion/references/runtime.md` for the `Agent`,
   `Environment`, `Task`, `Grader`, and `Orchestrator` model.
4. Use `.claude/skills/ac2-companion/references/datasets.md` before eval or training work.
5. Use `.claude/skills/ac2-companion/references/eval-local.md` and
   `.claude/skills/ac2-companion/references/eval-remote.md` for eval workflows.
6. Use `.claude/skills/ac2-companion/references/training.md` for training submissions.
7. Use `.claude/skills/ac2-companion/references/migrations.md` before porting another
   framework into AC2.

## Project Order

1. `rock-paper-scissors/` for local sessions, tools, grading, and deployment.
2. `dapo-math-check/` for datasets, custom user policy, eval, and training.
3. `tau2bench/` for simulated users, domain environments, remote eval, and training.

## Editing Guidelines

- Prefer existing AC2 primitives and local project patterns over new scaffolding.
- Keep remote eval and training instructions explicit about projects, datasets, secrets,
  commits, and pushed branches.
- Avoid internal-only launchers, monorepo paths, or infrastructure names in user-facing
  READMEs.
- After editing a project, run the narrowest relevant tests or syntax checks for that
  project and report what was run.
