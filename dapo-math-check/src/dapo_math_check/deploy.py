from __future__ import annotations

from ac2.sdk import Client, DeploymentConfig

CONFIG = DeploymentConfig(agent="DapoMathAgent", env="DapoMathCheckEnvironment")


def main() -> None:
    Client(project="dapo-math-check").deployments.create(CONFIG)


if __name__ == "__main__":
    main()
