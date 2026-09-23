from ac2.runtime import Environment, Grader, GraderOutput, Message, Trace
from math_verify import LatexExtractionConfig, parse, verify
from pydantic import BaseModel, JsonValue


class Answer(BaseModel):
    answer: str


def answer_score(response: str, expected: str) -> float:
    # AC2 invokes graders in worker threads, where signal timeouts are unavailable.
    gold = parse(f"${expected}$", extraction_config=[LatexExtractionConfig()], parsing_timeout=None)
    predicted = parse(response, extraction_config=[LatexExtractionConfig()], parsing_timeout=None)
    return float(bool(gold and predicted and verify(gold, predicted, timeout_seconds=None)))


class Geo3KGrader(Grader):
    async def _grade(self, grader_params: dict[str, JsonValue] | None, trace: Trace, env: Environment) -> GraderOutput:
        expected = Answer.model_validate(grader_params).answer
        items = trace[-1].get_items() if trace else []
        content = next(
            (item.content for item in reversed(items) if isinstance(item, Message) and item.role == "assistant"),
            "",
        )
        if isinstance(content, list):
            content = "\n".join(part["text"] for part in content if part.get("type") == "text")
        score = answer_score(content, expected)
        return GraderOutput(
            score=score, reasoning="Final answer matches." if score else "Final answer does not match."
        )
