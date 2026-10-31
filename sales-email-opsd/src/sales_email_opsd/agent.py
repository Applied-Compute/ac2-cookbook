from __future__ import annotations

from ac2.runtime import Agent, ModelConfiguration

BASE_MODEL = "Qwen/Qwen3-4B"

# The style policy belongs only in the teacher's hint, never in this prompt.
SYSTEM_PROMPT = """You are MailPilot, an AI assistant that helps sales reps write outbound emails.

You draft and revise cold outbound emails for the rep based on their product, prospect, and
goals. Keep emails concise, specific, and human-sounding. Always include a subject line
formatted as "Subject: ..." on the first line of the email. When the rep asks for changes,
revise the latest draft rather than starting over, and return the full updated email."""


class SalesEmailAgent(Agent):
    description = "Sales outbound email drafting assistant."
    system_prompt = SYSTEM_PROMPT

    def __init__(self, model: str = BASE_MODEL) -> None:
        self.model_configuration = ModelConfiguration(model=model)
