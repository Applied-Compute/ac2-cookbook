from __future__ import annotations

from ac2.runtime import Agent, EnvironmentProtocol, Message, ModelConfiguration

from .dataloader import load_tau2_policy


class Tau2AirlineAgent(Agent):
    description = "Airline customer service agent."
    model_configuration = ModelConfiguration(model="gpt-5.6-terra")

    def get_system_prompt(self, env: EnvironmentProtocol) -> Message:
        return Message(role="system", content=load_tau2_policy("airline"))


class Tau2RetailAgent(Agent):
    description = "Retail customer service agent."
    model_configuration = ModelConfiguration(model="gpt-5.6-terra")

    def get_system_prompt(self, env: EnvironmentProtocol) -> Message:
        return Message(role="system", content=load_tau2_policy("retail"))


class Tau2TelecomAgent(Agent):
    description = "Telecom customer service agent."
    model_configuration = ModelConfiguration(model="gpt-5.6-terra")

    def get_system_prompt(self, env: EnvironmentProtocol) -> Message:
        return Message(role="system", content=load_tau2_policy("telecom"))
