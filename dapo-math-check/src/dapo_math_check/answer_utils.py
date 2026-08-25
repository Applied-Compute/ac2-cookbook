import re


def normalize_answer(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()
    text = text.replace("$", "").replace(",", "").replace(" ", "")

    match = re.fullmatch(r"-?\d+\.?\d*", text)
    if match is None:
        return text.lower()

    try:
        num = float(text)
    except ValueError:
        return text.lower()
    if num == int(num):
        return str(int(num))
    return f"{num:.6f}".rstrip("0").rstrip(".")


def answers_match(given: str, expected: str) -> bool:
    return normalize_answer(given) == normalize_answer(expected)
