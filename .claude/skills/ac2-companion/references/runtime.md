# Defining agents, environments, graders, and tasks

This is the core of AC2: the runtime primitives the user defines once and reuses across local runs, evals, deployments, and training. Everything is in `ac2.runtime`. The same component classes feed `EvalConfig`, `TrainingConfig`, and `DeploymentConfig` — there is no separate "production" version.

## The mental model

Across local runs, evals, deployments, and training, the shape is the same:

```
        Task ──input──► Agent + Environment ──trace──► Grader ──score──►
```

- An **Agent** owns model configuration + system prompt + lifecycle hooks. It produces completions.
- An **Environment** owns runtime state and tools (async methods decorated with `@tool`). Tools run inside the env's setup/step/teardown lifecycle.
- A **Task** carries `input` (messages), optional `env_params` (passed to `env.setup`), and optional `grader_params` (passed to the grader).
- AC2 supplies the standard single-agent loop when a config names an agent and environment. Only subclass `OrchestratorProtocol` when the rollout needs multiple agents or custom control flow.
- A **Grader** scores the trace after the rollout and returns a `GraderOutput(score, reasoning, artifacts)`.

Tracing is automatic: AC2 runtime components (agents, environments, tools, orchestrators) are already instrumented with OpenTelemetry. Spans flow to ClickHouse and surface on the dashboard. Add `@traced` only to your own helper code when you want it in the span tree.

## Agent

An agent wraps an LLM and produces completions. Its registered name is its Python class name. Put static configuration on the class and use a no-argument constructor only for instance state.

```python
from ac2.runtime import Agent, ModelConfiguration


class AssistantAgent(Agent):
    description = "A helpful assistant."
    model_configuration = ModelConfiguration(model="gpt-4o-mini")
    system_prompt = "You are a helpful assistant."
```

| Class variable | Type | Description |
|----------------|------|-------------|
| `description` | `str` | Human-readable description |
| `model_configuration` | `ModelConfiguration \| None` | Model + provider settings |
| `system_prompt` | `str` | System prompt text |
| `allowed_tools` | `list[str] \| None` | Restrict which env tools this agent sees |

### Model configuration

```python
ModelConfiguration(
    model="gpt-4o-mini",                    # required
    kwargs={"temperature": 0.7},            # passed through to provider
)
```

| Field | Default | Description |
|-------|---------|-------------|
| `model` | — | Model name (e.g. `"gpt-4o"`, `"claude-sonnet-4-20250514"`) |
| `base_url` | `None` | Custom OpenAI-compatible endpoint |
| `api_type` | `"responses"` | OpenAI API format. Use `"completions"` for OpenAI-compatible chat APIs |
| `kwargs` | `{}` | Provider passthrough (temperature, reasoning, etc.) |

**Routing rule:** model names starting with `"claude"` route to Anthropic, everything else routes to OpenAI. Credentials come from standard env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

### Subclassing for hooks

Subclass `Agent` when you need lifecycle hooks. The built-in hooks fire at well-defined points in each turn:

```python
from ac2.runtime import Agent, Environment, Item, ModelConfiguration


class MyAgent(Agent):
    description = "A helpful assistant."
    model_configuration = ModelConfiguration(model="gpt-4o-mini")
    system_prompt = "You are a helpful assistant."

    async def on_completion(self, items: list[Item], env: Environment) -> None:
        print(f"Model produced {len(items)} items")

    async def on_env_step_error(self, error: Exception, env: Environment) -> None:
        print(f"Tool error: {error}")
```

| Hook | When | Default |
|------|------|---------|
| `get_system_prompt(env)` | Episode start | Returns the configured system prompt as a `Message` |
| `get_completion(items, env, stream)` | Each loop iteration | Calls `CompletionClient` with current items + env tools |
| `on_turn_start(env)` | Start of each turn | No-op |
| `on_completion(items, env)` | After a completion | No-op |
| `on_env_step(items, env)` | After env processes tool calls | No-op |
| `on_env_step_error(error, env)` | When `env.step()` raises | No-op |
| `on_turn_end(env)` | End of each turn | No-op |

For full control (e.g. replacing `CompletionClient` with a custom integration), subclass `AgentProtocol` instead of `Agent`.

## Environment

An environment owns state and tools. The base `Environment` is a no-op shell; subclass it to add tools and per-session state.

```python
import random
from typing import Annotated

from pydantic import Field

from ac2.runtime import Environment, tool


class WeatherEnvironment(Environment):
    @tool("Get the current weather for a city.")
    async def get_weather(
        self,
        city: Annotated[str, Field(description="The city name.")],
    ) -> str:
        return f"{city}: {random.choice(['sunny', 'cloudy', 'rainy'])}"
```

### Tools

- Decorate async methods with `@tool("description")`.
- Tool schemas are derived from type hints + `Annotated[..., Field(description=...)]` for parameter docs.
- The runtime serializes the signature to JSON schema and feeds it to the model.
- Return values are stringified; non-string returns are JSON-serialized.
- All tool calls in one model response are executed concurrently with `asyncio.gather`. If a tool raises, the runtime returns a `FunctionCallOutput` containing the error so the model can recover.

### Tool patterns

Four contrasting shapes that cover most real tools. Copy as starting points.

**Stateless tool with structured input.** No env state; the tool is pure.

```python
class CalcEnv(Environment):
    @tool("Add two integers.")
    async def add(
        self,
        a: Annotated[int, Field(description="First integer.")],
        b: Annotated[int, Field(description="Second integer.")],
    ) -> int:
        return a + b
```

**Tool that reads state populated by `setup(env_params)`.** Per-rollout config (a path, a ground-truth answer, a tenant ID) goes in `Task.env_params`; `setup` stashes it on `self`; tools read it.

```python
class AnswerCheckEnv(Environment):
    expected: str = ""

    async def setup(self, env_params: dict | None = None) -> None:
        self.expected = (env_params or {}).get("expected", "")

    @tool("Submit your final answer; returns whether it matches.")
    async def submit(
        self,
        answer: Annotated[str, Field(description="The final answer.")],
    ) -> str:
        return "correct" if answer.strip() == self.expected else "incorrect"
```

**Tool that raises.** Don't catch errors and stringify them yourself — `raise` and let the runtime convert the exception into a `FunctionCallOutput` the model can read. The model sees the error and can retry with different arguments.

```python
@tool("Look up a record by id.")
async def lookup(
    self,
    record_id: Annotated[str, Field(description="Record id.")],
) -> dict:
    record = await self._store.get(record_id)
    if record is None:
        raise KeyError(f"No record with id={record_id!r}")
    return record  # dicts are JSON-serialized for the model
```

**Tool with an optional/defaulted parameter.** `Annotated[..., Field(...)]` carries the description and validation; the Python default carries the value.

```python
@tool("List recent records, newest first.")
async def list_records(
    self,
    limit: Annotated[int, Field(description="Max records.", ge=1, le=100)] = 10,
) -> list[dict]:
    return await self._store.list_recent(limit)
```

### Lifecycle

| Method | When | Default |
|--------|------|---------|
| `setup(env_params)` | Once, before the first turn | No-op |
| `step(items)` | After each model completion | Finds `FunctionCall` items, runs matching tools concurrently, returns `(outputs, done)`. `done=True` when there are no function calls. |
| `teardown()` | Once, after the session ends | No-op |

`env_params` is a dict from the `Task` (or session caller) — use it for per-rollout config like tenant IDs, problem-specific answers, or feature flags.

```python
class DatabaseEnvironment(Environment):
    db: DatabaseConnection | None = None

    async def setup(self, env_params: dict) -> None:
        self.db = await connect(env_params.get("database_url", ""))

    async def teardown(self) -> None:
        if self.db:
            await self.db.close()
```

### Overriding `step`

Override `step` when you need custom behavior — logging, rate limiting, post-processing tool outputs, or **sequential** execution (the default is concurrent, which can break stateful tools that depend on per-call feedback).

```python
async def step(self, items: list[Item]) -> tuple[list[FunctionCallOutput], bool]:
    outputs, done = await super().step(items)
    for output in outputs:
        print(f"Tool returned: {output.output[:100]}", flush=True)
    return outputs, done
```

Return value `(outputs, done)` controls whether the agent loop continues. `done=True` ends the turn.

### Restricting tools per agent

`allowed_tools` on the agent filters which tool schemas the model sees. Useful when one env defines many tools but a given agent should only use a subset:

```python
class Researcher(Agent):
    description = "Finds information."
    model_configuration = ModelConfiguration(model="gpt-4o-mini")
    system_prompt = "Use browser_search to find information."
    allowed_tools = ["browser_search"]
```

## Grader

A grader scores the trace after the rollout completes and returns a `GraderOutput(score, reasoning, artifacts)`.

```python
from ac2.runtime import Environment, Grader, GraderOutput, Message, Trace


class SubstringGrader(Grader):
    async def _grade(
        self,
        grader_params: dict | None,
        trace: Trace,
        env: Environment,
    ) -> GraderOutput:
        expected = (grader_params or {}).get("expected_answer", "")
        items = trace[-1].get_items() if trace else []
        answer = next(
            (item.content for item in reversed(items)
             if isinstance(item, Message) and item.role == "assistant"),
            "",
        )
        match = expected.lower() in str(answer).lower()
        return GraderOutput(
            score=1.0 if match else 0.0,
            reasoning=f"Expected {expected!r}.",
        )
```

Implement `_grade(grader_params, trace, env) -> GraderOutput`. The public `grade(...)` wraps it with tracing and score emission.

- `grader_params` comes from `Task.grader_params`.
- `trace` is the list of `Episode` objects the orchestrator produced; usually you care about `trace[-1].get_items()`.
- `env` is the environment after the rollout — graders that need env-internal state (terminal flags, accumulated counts) read it here.

### LLM-as-judge

Use `LLMGrader` instead of `Grader` to get a built-in `CompletionClient`:

```python
from ac2.runtime import LLMGrader, ModelConfiguration


class JudgeGrader(LLMGrader):
    model_config = ModelConfiguration(model="gpt-4o", kwargs={"temperature": 0})

    async def _grade(self, grader_params, trace, env) -> GraderOutput:
        # ... build messages, call self.get_completion(items), parse output ...
```

## Task

A `Task` is one rollout target. It carries `input` (the initial messages), optional `env_params` (passed to `env.setup`), and optional `grader_params` (passed to the grader).

```python
from ac2.runtime import Message, Task

task = Task(
    input=[Message(role="user", content="What is the capital of France?")],
    env_params={"hint": "European country"},
    grader_params={"expected_answer": "Paris"},
)
```

Tasks are passed to evals either inline (`tasks=[...]`) or via a dataset uploaded to the platform (see `eval-local.md` and `eval-remote.md`).

## Orchestrator: canonical vs. custom

For a conventional single-agent loop, you don't need a custom orchestrator. Name the agent and environment directly in the config:

```python
config = EvalConfig(agent="AssistantAgent", env="WeatherEnvironment", ...)
```

Build a custom orchestrator when the rollout needs multiple agents or custom control flow (judges picking between drafts, voting, fork-and-merge, planner-writer-reviewer chains, etc.).

### Subclass `OrchestratorProtocol`

The protocol provides default `setup`, `teardown`, `session`, `trace`, and a class-name-based `name`. A subclass constructs its public `env` and `agents` in a no-argument constructor and implements `run`.

```python
from collections.abc import AsyncIterator

from ac2.runtime import (
    Agent,
    Environment,
    Input,
    Message,
    ModelConfiguration,
    OrchestratorProtocol,
)
from ac2.runtime.orchestration.streams.types import StreamEvent


class DirectDrafter(Agent):
    description = "Writes a direct, concise answer."
    model_configuration = ModelConfiguration(model="gpt-4o-mini")
    system_prompt = "Answer the question directly and concisely."


class DetailedDrafter(Agent):
    description = "Writes a detailed, thorough answer."
    model_configuration = ModelConfiguration(model="gpt-4o-mini")
    system_prompt = "Answer the question thoroughly with examples."


class Judge(Agent):
    description = "Picks the better draft."
    model_configuration = ModelConfiguration(model="gpt-4o-mini")
    system_prompt = (
        "You will receive two drafts labeled A and B. "
        "Pick the better one and respond with only that draft's text, unchanged."
    )


class TournamentEnvironment(Environment):
    pass


class TournamentOrchestrator(OrchestratorProtocol):
    def __init__(self) -> None:
        self.env = TournamentEnvironment()
        self.drafter_a = DirectDrafter()
        self.drafter_b = DetailedDrafter()
        self.judge = Judge()
        self.agents = [self.drafter_a, self.drafter_b, self.judge]

    async def run(self, turn_input: Input) -> AsyncIterator[StreamEvent]:
        ep_a = self.session.start_episode(self.drafter_a)
        async for event in self.session.run_turn(ep_a, turn_input):
            pass

        ep_b = self.session.start_episode(self.drafter_b)
        async for event in self.session.run_turn(ep_b, turn_input):
            pass

        draft_a = _last_assistant_text(ep_a)
        draft_b = _last_assistant_text(ep_b)

        judge_ep = self.session.start_episode(self.judge)
        async for event in self.session.run_turn(
            judge_ep,
            [Message(role="user", content=f"Draft A:\n{draft_a}\n\nDraft B:\n{draft_b}")],
        ):
            yield event


def _last_assistant_text(episode) -> str:
    for item in reversed(episode.get_items()):
        if isinstance(item, Message) and item.role == "assistant" and item.content:
            return item.content
    return ""
```

The pattern: start an `Episode` per agent via `self.session.start_episode(agent)`, drive each agent's turn with `self.session.run_turn(episode, turn_input)`, and `yield` events only from the turn(s) whose output should reach the caller. Earlier turns can be consumed with `async for _ in ...: pass` so the caller only sees the final agent's stream.

## What goes where: a checklist

When the user is writing their first agent, walk through this:

1. **Agent**: subclass with model + system prompt class variables; add hooks only when needed.
2. **Environment**: tools the agent can call, any per-rollout state in `setup(env_params)`. Override `step` only if concurrent tool dispatch is wrong for their case.
3. **Task / dataset**: each rollout's `input` + `env_params` + `grader_params`. Tasks can be inline for quick experiments, uploaded as a dataset once they stabilize.
4. **Grader**: reads the final trace (and optionally env state) and emits a score. Substring match for exact answers, `LLMGrader` for judge-style scoring, env-state read for graders that depend on env terminal flags.
5. **Orchestrator**: nothing to define for single-agent loops — put the agent and environment class names in the config. Build a custom `OrchestratorProtocol` subclass only for multi-agent or custom-control-flow rollouts.

Once these are in place, their exact class names feed into `EvalConfig`, `DeploymentConfig`, and `TrainingConfig`.

## Next

- Run an eval against these primitives: `references/eval-local.md`.
- Deploy the agent for interactive use: `references/deploy.md`.
- Train against them: `references/training.md`.
