from __future__ import annotations

from ac2.runtime import Agent, EnvironmentProtocol, Message, ModelConfiguration

from .dataloader import Domain, load_tau2_policy

DOMAIN_ENVIRONMENT: dict[Domain, str] = {
    "airline": "AirlineEnvironment",
    "retail": "RetailEnvironment",
    "telecom": "TelecomEnvironment",
}


class Tau2Agent(Agent):
    """Customer-service agent; `domain` selects the tau2-bench policy."""

    model_configuration = ModelConfiguration(model="gpt-5.6-terra")

    def __init__(self, domain: Domain = "airline") -> None:
        self.domain = domain
        self.description = f"{domain.capitalize()} customer service agent."

    def get_system_prompt(self, env: EnvironmentProtocol) -> Message:
        return Message(role="system", content=load_tau2_policy(self.domain))
