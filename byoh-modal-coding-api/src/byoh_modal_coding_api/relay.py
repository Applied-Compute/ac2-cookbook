from ac2.sdk import Client

RELAY_NAME = "custom-harness-relay"


def running_relay_id(client: Client) -> str:
    for deployment in client.app_deployments.list().items:
        if deployment.name == RELAY_NAME and deployment.state == "running":
            return deployment.id
    raise RuntimeError(
        "Model access is not available for this BYOH project. "
        "Please contact the AC team to check the project setup."
    )
