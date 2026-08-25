# Migrating agent frameworks to AC2

Use this guide before porting code from Claude Agents SDK, OpenAI Agents SDK, Prime Verifiers, Silverback, or a custom harness. The goal is not drop-in compatibility. The goal is to preserve the task contract, then express it with AC2-native primitives.

## Migration order

1. Write the source contract down before coding: input messages, system/developer prompts, tools, tool schemas, state, stop conditions, scoring, datasets, and metrics.
2. Decide which AC2 primitive owns each piece.
3. Port the smallest runnable slice and run it locally.
4. Add dataset upload only after inline/local tasks work.
5. Add remote eval, deployment, or training only after local behavior and grading are stable.

## Concept mapping

| Source construct | AC2 construct | Notes |
|---|---|---|
| Model wrapper, assistant, policy, system prompt | `Agent` subclass | Put model configuration and prompts on class variables. Avoid overriding `get_completion` unless you know training compatibility is irrelevant. |
| Tool definitions | `Environment` methods decorated with `@tool` | AC2 tools are async methods. Use type hints plus `Annotated[..., Field(description=...)]` for schemas. |
| Per-rollout state, sandbox/session handles, task-specific config | `Environment.setup(env_params)` and env instance fields | Put values that vary per task in `Task.env_params`. Close resources in `teardown`. |
| Dataset row, eval sample, benchmark instance | `Task` | `Task.input` is the initial message list. `env_params` feeds the environment. `grader_params` feeds the grader. |
| Reward function, scorer, evaluator, rubric | `Grader` or `LLMGrader` | Deterministic checks should subclass `Grader`; judge-based scoring can subclass `LLMGrader`. |
| Multi-agent flow, planner/writer/reviewer, judge selection, compaction | `OrchestratorProtocol` | Start episodes with `self.session.start_episode(agent)` and drive turns with `self.session.run_turn(...)`. |
| User simulator, nudges after plain-text replies, forced tool loops | `User` policy, or orchestrator-level loop | Do not make the environment emit user messages; `Environment.step` returns tool outputs only. |
| Local eval runner | `EvalConfig` + `client.eval.run(config, local=True)` | Use inline `tasks=[...]` while iterating, or `dataset="..."` once the task set stabilizes. |
| Remote eval runner | `EvalConfig` + `client.eval.run(config)` | Requires a dataset name; AC2 builds/uploads the project. Inline tasks are local-only. |
| Hosted training config | `TrainingConfig` + `client.train.run(config)` | Reuses the same named components, grader, and datasets as eval. |
| Deployment/session endpoint | `DeploymentConfig` + `client.deployments.create(config)` | Use for interactive serving, not for eval-only workflows. |

## Claude Agents SDK

Claude-style agents often combine instructions, tools, subagents, and stop behavior in one agent definition. In AC2, split those responsibilities:

- Put the assistant instructions and model selection in `Agent`.
- Put tools and any tool-visible world state in `Environment`.
- Put subagent delegation behind a normal environment tool if the main agent should observe only a result.
- Put multi-agent peer workflows in `OrchestratorProtocol` if each agent should have its own episode in the trace.
- Put forced-tool nudges and simulated user turns in `User` or the orchestrator, not in the environment.

If the source system mutates or compacts previous messages, represent that explicitly by opening a new AC2 episode with copied or summarized context. Do not add mutable trace operations to AC2 episodes.

## OpenAI Agents SDK

OpenAI Agents SDK ports usually start with an agent, handoff targets, tools, and guardrails.

- Agent instructions and model settings map to `Agent`.
- Function tools map to `@tool` methods.
- Handoffs that are just helper calls can become a tool that calls a subagent and returns text.
- Handoffs between peer agents should become an `OrchestratorProtocol` flow with separate episodes.
- Input/output guardrails usually map to a `Grader`, an environment tool validation error, or an orchestrator stop condition depending on when the check must run.

Do not preserve SDK-specific control objects if AC2 has a simpler native boundary. Preserve observable behavior: what the model sees, which actions it can take, when the rollout stops, and how it is scored.

## Prime Verifiers

Prime Verifiers centers environment packages around `load_environment(...)`, datasets, rubrics, and environment classes such as `SingleTurnEnv`, `ToolEnv`, `StatefulToolEnv`, and `MultiTurnEnv`.

The closest AC2 mapping is:

- Verifiers dataset rows -> AC2 `Task` objects.
- Verifiers rubric/reward functions -> AC2 `Grader`.
- `ToolEnv` / `StatefulToolEnv` tools -> AC2 `Environment` with `@tool` methods and `setup(env_params)`.
- `MultiTurnEnv.env_response(...)` -> usually an AC2 `User` policy or `OrchestratorProtocol` loop.
- Verifiers eval command -> a runnable module that calls `client.eval.run(config, local=...)`.
- Prime hosted training config -> AC2 `TrainingConfig`.

Be careful with rollout ownership. In Verifiers, the environment may own more of the conversation loop. In AC2, the environment should not emit user messages. It processes tool calls and returns `FunctionCallOutput` items. Conversation control lives above it.

## Silverback / custom RL harnesses

Silverback-style environments often combine task loading, tool execution, user nudges, stop criteria, and grading in one class. When porting:

- Move task loading into `Task` creation and dataset registration scripts.
- Move tool execution into `Environment`.
- Move per-task answers or constraints into `env_params` and `grader_params`.
- Move forced-tool nudges into a custom `User`.
- Move multi-agent or multi-phase control into `OrchestratorProtocol`.
- Move scoring into `Grader`.

`dapo-math-check` is the in-repo example of this split: the environment is a bounded `check_answer` tool with termination flags, the user policy nudges the agent across turns, and the grader reads the final environment state.

## Layering rules

### Keep platform operations at the SDK boundary

Use the SDK for platform-facing setup: projects, datasets, secrets, evals, deployments, and training submissions. Keep agent, environment, and grader code focused on the behavior being evaluated or served.

### Environments own tools

An environment should own:

- `@tool` methods.
- Per-rollout resource setup and teardown.
- Tool-visible state.
- Termination flags that graders or users can inspect.

An environment should not own:

- Multi-agent sequencing.
- User messages or nudges.
- Dataset creation or upload.
- Training submission or platform registration.

### Orchestrators own control flow

Put an agent class name and environment class name directly in the config for the common single-agent loop. Use `OrchestratorProtocol` only when you need multiple agents, multiple episodes, a judge choosing between drafts, compaction, fork/merge behavior, or custom turn sequencing.

### Graders own scoring

Keep reward logic in a `Grader` even if the source framework scored inside the environment. The grader receives the final trace and environment, so it can inspect both conversation items and environment state without mixing scoring into tool execution.

## Porting checklist

- Source prompts and role ordering are documented.
- Every source tool has an AC2 `@tool` method with explicit type hints and descriptions.
- Per-task data is divided between `Task.input`, `env_params`, and `grader_params`.
- Stop conditions are assigned to the environment, `User`, or orchestrator intentionally.
- Scoring returns `GraderOutput(score=..., reasoning=..., artifacts=...)`.
- Local session or local eval passes before remote runs.
- Remote workflows use a managed project (`ac2 project init`) with exactly one package under `src/` and runtime deps in `[project].dependencies`.
- Provider secrets are uploaded before remote eval, deployment, or training.
- Training config names the correct trainable agent when the orchestrator has multiple agents.
