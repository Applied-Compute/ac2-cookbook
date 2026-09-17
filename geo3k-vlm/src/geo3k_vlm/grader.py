from ac2.runtime import Grader, GraderOutput, Message
from math_verify import LatexExtractionConfig, parse, verify


def answer_score(response: str, expected: str) -> float:
    # Dispatch invokes graders in worker threads, where signal timeouts are unavailable.
    gold = parse(f"${expected}$", extraction_config=[LatexExtractionConfig()], parsing_timeout=None)
    predicted = parse(response, extraction_config=[LatexExtractionConfig()], parsing_timeout=None)
    return float(bool(gold and predicted and verify(gold, predicted, timeout_seconds=None)))


class Geo3KGrader(Grader):
    async def _grade(self, grader_params, trace, env) -> GraderOutput:
        items = trace[-1].get_items() if trace else []
        content = next((item.content for item in reversed(items)
                        if isinstance(item, Message) and item.role == "assistant"), "")
        if isinstance(content, list):
            content = "\n".join(part["text"] for part in content if part.get("type") == "text")
        score = answer_score(content or "", grader_params["answer"])
        return GraderOutput(score=score, reasoning="Final answer matches." if score else "Final answer does not match.")
