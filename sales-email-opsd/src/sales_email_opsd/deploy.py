from __future__ import annotations

import argparse

from ac2.sdk import Client, DeploymentConfig

from .agent import BASE_MODEL, SalesEmailAgent
from .environment import SalesEmailEnvironment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy the email agent using a provider model or served endpoint."
    )
    parser.add_argument("--project", default="sales-email-opsd")
    parser.add_argument(
        "--model", required=True, help=f"A reachable endpoint, not an unserved slug such as {BASE_MODEL}"
    )
    args = parser.parse_args()
    config = DeploymentConfig(agent=SalesEmailAgent(model=args.model), env=SalesEmailEnvironment())
    print(Client(project=args.project).deployments.create(config))


if __name__ == "__main__":
    main()
