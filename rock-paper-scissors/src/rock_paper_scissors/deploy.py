from __future__ import annotations

from ac2.sdk import Client, DeploymentConfig

CONFIG = DeploymentConfig(agent="RPSAgent", env="RPSEnvironment")


def main() -> None:
    Client(project="rock-paper-scissors").deployments.create(CONFIG)


if __name__ == "__main__":
    main()
