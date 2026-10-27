"""The executable style policy shared by dataset selection and grading."""

from __future__ import annotations

import re
from dataclasses import dataclass

EM_DASH = "—"
POLICY_VERSION = "dash-punctuation-v2"
POLICY = (
    "Do not use em dashes or dash punctuation in sales emails. Replace sentence breaks "
    "with a comma, period, colon, or a rewritten sentence. Keep ordinary hyphenated "
    "compounds, numeric ranges, and list markers. Preserve the email's meaning, facts, "
    "subject line, and call to action."
)
_CANDIDATE = re.compile(r"—|--|–|(?<=[^\S\n])-\s")


@dataclass(frozen=True)
class DashViolation:
    start: int
    end: int
    excerpt: str


def find_dashes(text: str) -> list[DashViolation]:
    violations = []
    for match in _CANDIDATE.finditer(text):
        start, end = match.span()
        if match.group() != EM_DASH:
            line_start = text.rfind("\n", 0, start) + 1
            line_end = text.find("\n", end)
            left = text[line_start:start].rstrip()
            right = text[end : line_end if line_end >= 0 else len(text)].lstrip()
            if not left or not right:
                continue
            if left[-1].isdigit() and right[0].isdigit():
                continue
            if not any(char.isalnum() for char in left) or not any(char.isalnum() for char in right):
                continue
        excerpt = text[max(0, start - 45) : min(len(text), end + 45)].replace("\n", " ")
        violations.append(DashViolation(start=start, end=end, excerpt=excerpt))
    return violations


def count_dashes(text: str) -> int:
    return len(find_dashes(text))
