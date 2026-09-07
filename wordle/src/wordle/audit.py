"""Check exported AC2 traces for actual game completion, independent of reward."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(traces) -> dict:
    outcomes = Counter()
    token_limit_hits = 0
    for trace in traces:
        grades = [s for s in trace.get("scores", []) if s.get("grader_name") == "WordleGrader"]
        if len(grades) != 1:
            outcomes["missing_or_ambiguous_grade"] += 1
        else:
            artifacts = grades[0].get("artifacts") or {}
            if isinstance(artifacts, str):
                artifacts = json.loads(artifacts)
            reason = artifacts.get("terminated_reason") or "unfinished"
            guesses = artifacts.get("guesses", 0)
            if reason == "correct" and artifacts.get("solved") and 1 <= guesses <= 6:
                outcomes["correct"] += 1
            elif reason == "exhausted" and not artifacts.get("solved") and guesses == 6:
                outcomes["exhausted"] += 1
            else:
                outcomes[reason if reason in {"unfinished", "abandoned"} else "invalid_terminal_state"] += 1
        errors = " ".join(s.get("error_message") or "" for s in trace.get("spans", []))
        token_limit_hits += "finish_reason=length" in errors or "budget exhausted" in errors.lower()
    total = sum(outcomes.values())
    completed = outcomes["correct"] + outcomes["exhausted"]
    return {
        "rollouts": total, "completed": completed,
        "completion_rate": completed / total if total else 0,
        "token_limit_hits": token_limit_hits, "outcomes": dict(outcomes),
        "passed": total > 0 and completed == total and token_limit_hits == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, help="JSONL file from client.traces.download().")
    args = parser.parse_args()
    with args.traces.open() as source:
        result = audit(json.loads(line) for line in source if line.strip())
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
