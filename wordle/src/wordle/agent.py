import os

from ac2.runtime import Agent, ModelConfiguration

SYSTEM_PROMPT = """Play Wordle: discover the hidden five-letter English word.
Use check_answer(guess) to submit one guess at a time, then use its feedback:
🟩 = correct letter and position; 🟨 = correct letter, wrong position;
⬜ = no unmatched copy of this letter remains in the answer.
Green matches are assigned first. Yellow matches use remaining copies of each
letter, from left to right. The answer may contain repeated letters.
Guesses must be five ASCII letters and in the dictionary. Surrounding whitespace
is stripped and lowercase is accepted. Invalid guesses return "invalid guess".
You have six attempts, including invalid submissions. A correct guess ends the
game immediately. A win on attempt k earns (11-k)/10; losing earns zero.
Use the feedback to choose a new word; do not repeat failed guesses. Keep your
reasoning focused, then call check_answer exactly once. Continue until the game
ends. Submit guesses through the tool and do not end with a final text answer."""

JSON_TOOL_PROMPT = """Wrap every tool call in <tool_call> and </tool_call> tags. For example:
<tool_call>
{"name": "check_answer", "arguments": {"guess": "crane"}}
</tool_call>
Use that format with your chosen word on every turn; bare JSON is not a tool call."""


EVAL_MODEL = os.environ.get("WORDLE_EVAL_MODEL", "Qwen/Qwen3-4B")


class WordleAgent(Agent):
    description = "Plays Wordle using only check_answer."
    model_configuration = ModelConfiguration(
        model=EVAL_MODEL, api_type="completions",
        base_url=os.environ.get("WORDLE_EVAL_BASE_URL"),
        api_key_env=os.environ.get("WORDLE_EVAL_API_KEY_ENV"),
        kwargs={
            "max_tokens": 8192, "temperature": 1.0,
            **({"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
               if "qwen3" in EVAL_MODEL.lower() else {}),
        },
    )
    allowed_tools = ["check_answer"]
    system_prompt = SYSTEM_PROMPT + "\n" + JSON_TOOL_PROMPT


class WordleQwen36Agent(WordleAgent):
    """Use the XML calls required by Qwen3.6's qwen3_coder parser."""

    model_configuration = ModelConfiguration(
        model="Qwen/Qwen3.6-35B-A3B", api_type="completions",
        kwargs={"max_tokens": 8192, "temperature": 1.0,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    )
    system_prompt = SYSTEM_PROMPT + """
Wrap every tool call in the model's XML function format. For example:
<tool_call>
<function=check_answer>
<parameter=guess>
crane
</parameter>
</function>
</tool_call>
Use that format with your chosen word on every turn."""
