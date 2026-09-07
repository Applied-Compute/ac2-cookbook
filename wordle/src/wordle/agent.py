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
Keep reasoning brief. Submit your answer with the tool, not as plain text."""


class WordleAgent(Agent):
    description = "Plays Wordle using only check_answer."
    model_configuration = ModelConfiguration(
        model=os.environ.get("WORDLE_EVAL_MODEL", "Qwen/Qwen3-4B"), api_type="completions",
        base_url=os.environ.get("WORDLE_EVAL_BASE_URL"),
        api_key_env=os.environ.get("WORDLE_EVAL_API_KEY_ENV"),
        kwargs={"max_tokens": 1024, "temperature": 1.0},
    )
    allowed_tools = ["check_answer"]
    system_prompt = SYSTEM_PROMPT
