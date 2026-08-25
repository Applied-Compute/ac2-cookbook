# Deploying an agent and connecting to it

Define a `DeploymentConfig` with component class names and pass it to the SDK:

```bash
uv run python -m my_project.deploy
```

SDK equivalent from inside the managed project:

```python
deployment_id = Client(project="my-project").deployments.create(CONFIG)
```

## Local session first

```python
import asyncio

from ac2.sdk import Client

from my_project import MyAgent, MyEnvironment

client = Client(project="my-project")
session = client.session.from_local(agent=MyAgent(), env=MyEnvironment())


async def main() -> None:
    async with session:
        async for event in session.run("Hello!"):
            if event.type == "text_delta":
                print(event.content, end="", flush=True)
        print()


asyncio.run(main())
```

## Config module

```python
# src/my_project/deploy.py
from ac2.sdk import Client, DeploymentConfig

CONFIG = DeploymentConfig(
    agent="MyAgent",
    env="MyEnvironment",
    cluster_id="ac-jurassic",  # optional
    tags={"team": "demo"},     # optional
)


def main() -> None:
    Client(project="my-project").deployments.create(CONFIG)
```

Pass `orchestrator=` instead of `agent` + `env` for custom control flow.

## Deploy

```bash
ac2 secrets put --key OPENAI_API_KEY --value <val>
uv run python -m my_project.deploy
```

AC2 builds the managed project from `[project].dependencies`, uploads it, and
resolves the named components in the deployment config.

## Connect remotely

```python
session = client.session.from_remote(deployment_id)

async with session:
    async for event in session.run("Hello!"):
        if event.type == "text_delta":
            print(event.content, end="", flush=True)
```

## Monitor

```bash
ac2 deployments list
ac2 deployments get <deployment_id>
ac2 deployments logs <deployment_id>
ac2 deployments stop <deployment_id>
ac2 deployments delete <deployment_id>
```
