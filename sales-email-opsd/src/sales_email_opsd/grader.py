from __future__ import annotations

import re

from ac2.runtime import EnvironmentProtocol, Grader, GraderOutput, Trace
from pydantic import JsonValue

from .dashes import EM_DASH, POLICY_VERSION, count_dashes
from .text import assistant_texts

_WORD = re.compile(r"\b\w+(?:[-']\w+)*\b")


def has_email_shape(text: str) -> bool:
    lines = text.strip().splitlines()
    if not lines:
        return False
    subject = lines[0].strip().strip("*# ")
    return (
        subject.lower().startswith("subject:")
        and bool(subject.partition(":")[2].strip())
        and any(line.strip() for line in lines[1:])
    )


class EmDashGrader(Grader):
    """Score the share of the model's visible replies that use no dash punctuation.

    Partial credit per reply shows progress that a whole-conversation pass hides, and a
    longer reply cannot dilute a dash. Email shape and the all-replies pass are diagnostics.
    """

    async def _grade(
        self,
        grader_params: dict[str, JsonValue] | None,
        trace: Trace,
        env: EnvironmentProtocol,
    ) -> GraderOutput:
        texts = assistant_texts(trace)
        counts = [count_dashes(text) for text in texts]
        dash_count = sum(counts)
        words = sum(len(_WORD.findall(text)) for text in texts)
        clean_messages = sum(bool(text.strip()) and count == 0 for text, count in zip(texts, counts))
        email_messages = sum(has_email_shape(text) for text in texts)
        truncated = any(episode.truncated for episode in trace)
        valid = bool(texts) and email_messages == len(texts) and not truncated
        clean = valid and dash_count == 0
        clean_message_fraction = clean_messages / len(texts) if texts else 0.0
        return GraderOutput(
            score=clean_message_fraction,
            reasoning=(
                f"{clean_messages} of {len(texts)} assistant message(s) use no dash punctuation; "
                f"{dash_count} violation(s) in total, {email_messages} message(s) contain a subject and body."
            ),
            artifacts={
                "policy_version": POLICY_VERSION,
                "dash_count": dash_count,
                "em_dash_count": sum(text.count(EM_DASH) for text in texts),
                "word_count": words,
                "dashes_per_1k_words": round(1000 * dash_count / words, 3) if words else None,
                "n_assistant_messages": len(texts),
                "clean_message_fraction": clean_message_fraction,
                "email_message_fraction": email_messages / len(texts) if texts else 0.0,
                "valid_output": valid,
                "truncated": truncated,
                "clean": clean,
            },
        )
