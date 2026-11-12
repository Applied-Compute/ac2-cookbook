"""Teacher-only guidance; never include these instructions in student inputs."""

JUDGE_PROMPT = """You are auditing an AI email assistant that drafts outbound sales emails for a sales rep.
The failure to catch: the assistant uses dashes as punctuation in its email drafts — an em
dash (—), or a hyphen, double hyphen, or en dash used between words (" - ", " -- ", " – ").
Hyphenated compound words like "attack-surface-management" or "20-minute" are fine and are
NOT the failure.

The assistant's draft:
{reply}

Write a teacher-only hint of at most 120 words that: tells the assistant never to use em
dashes or dash-style punctuation in sales emails; quotes 1-3 exact phrases from the draft
where such a dash appears (skip this if the draft has none); and rewrites each quoted
phrase without the dash, using a comma, period, colon, or a restructured sentence. Do not
flag hyphenated compound words. Reply with the hint only.
"""
