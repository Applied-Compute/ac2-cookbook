import random

from ac2.runtime import Message, Task

from .words import dictionary

PROJECT = "wordle"
TRAIN_DATASET = "wordle-train2048"
EVAL_DATASET = "wordle-eval256"


def make_task(base_word: str) -> Task:
    return Task(
        input=[Message(role="user", content="Find the hidden five-letter word. Make your first guess.")],
        env_params={"base_word": base_word},
    )


def build_datasets() -> dict[str, list[Task]]:
    words = sorted(dictionary())
    random.Random(42).shuffle(words)
    return {
        TRAIN_DATASET: [make_task(word) for word in words[:2048]],
        EVAL_DATASET: [make_task(word) for word in words[2048:2304]],
    }
