from ac2.runtime import Agent


class Geo3KAgent(Agent):
    description = "Solve geometry problems using the supplied diagram."
    allowed_tools = []
    system_prompt = (
        "Solve the geometry problem using the image. Show your reasoning, "
        r"then put your final answer in \boxed{...}."
    )
