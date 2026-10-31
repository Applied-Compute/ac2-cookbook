"""Teacher-only guidance; never include these instructions in student inputs."""

JUDGE_PROMPT = """You are auditing an AI email assistant that drafts outbound sales emails for a sales rep.
The failure to catch: the assistant uses dashes as punctuation in its email drafts — an em
dash (—), or a hyphen, double hyphen, or en dash used between words (" - ", " -- ", " – ").
Hyphenated compound words like "attack-surface-management" or "20-minute" are fine and are
NOT the failure.

Transcript ({n_assistant_turns} assistant turns; grader reward {reward}):
{transcript}

Grader reasoning: {grader_reasoning}

Call emit_online_hint_selection with the FIRST assistant turn whose draft uses a dash as
punctuation (or, if none does, the main email-drafting turn), and a teacher-only hint of
at most 120 words that: tells the assistant never to use em dashes or dash-style
punctuation in sales emails; quotes 1-3 exact phrases from that turn where such a dash
appears (skip this if the turn has none); and rewrites each quoted phrase without the
dash, using a comma, period, colon, or a restructured sentence. Do not flag hyphenated
compound words.
"""
