"""Build replay, online, and held-out tasks from a single conversation split."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ac2.runtime import HintMutation, HintSelection, Message, Task

from .agent import SYSTEM_PROMPT
from .dashes import POLICY, POLICY_VERSION, find_dashes
from .data import Conversation

TRAIN_DATASET = "sales-email-opsd-train-v2"
ONLINE_TRAIN_DATASET = "sales-email-opsd-online-train-v2"
EVAL_DATASET = "sales-email-eval-v2"


@dataclass(frozen=True)
class TaskSets:
    one_step: list[Task]
    online: list[Task]
    evaluation: list[Task]


def pick_boundaries(record: Conversation, maximum: int) -> list[int]:
    if maximum < 1:
        raise ValueError("max_boundaries must be positive")
    candidates = [
        i
        for i, message in enumerate(record.messages)
        if message.role == "assistant" and find_dashes(message.content)
    ]
    if len(candidates) <= maximum:
        return candidates
    if maximum == 1:
        return candidates[:1]
    return [candidates[round(i * (len(candidates) - 1) / (maximum - 1))] for i in range(maximum)]


def make_hint(reply: str) -> str:
    excerpts = [violation.excerpt for violation in find_dashes(reply)[:3]]
    if not excerpts:
        raise ValueError("A replay hint requires an offending assistant reply")
    return f"{POLICY}\nRevise the dash punctuation in these excerpts from your reply:\n" + "\n".join(
        f"- {json.dumps(excerpt, ensure_ascii=False)}" for excerpt in excerpts
    )


def replay_task(record: Conversation, boundary: int) -> Task:
    target = record.messages[boundary]
    if target.role != "assistant":
        raise ValueError("The replay boundary must point to an assistant message")
    return Task(
        input=[
            Message(role="system", content=SYSTEM_PROMPT),
            *(Message(role=m.role, content=m.content) for m in record.messages[: boundary + 1]),
        ],
        grader_params={
            "opsd_hint_selections": [
                HintSelection(
                    hint=make_hint(target.content),
                    rollout_prefix_len=boundary + 1,
                    mutation=HintMutation(item_boundary=boundary + 1, action="append_to_previous_item"),
                ).model_dump(mode="json")
            ]
        },
        tags={
            "conversation_id": record.conversation_id,
            "split": record.split,
            "boundary": boundary,
            "policy_version": POLICY_VERSION,
        },
    )


def conversation_task(record: Conversation) -> Task:
    opening = record.messages[0]
    return Task(
        input=[Message(role="user", content=opening.content)],
        env_params={
            "persona": record.persona.model_dump(),
            "followups": [m.content for m in record.messages[2:] if m.role == "user"],
        },
        grader_params={},
        tags={
            "conversation_id": record.conversation_id,
            "split": record.split,
            "policy_version": POLICY_VERSION,
        },
    )


def build_tasks(records: list[Conversation], *, max_boundaries: int = 3) -> TaskSets:
    if max_boundaries < 1:
        raise ValueError("max_boundaries must be positive")
    train = [record for record in records if record.split == "train"]
    held_out = [record for record in records if record.split == "eval"]
    if not train or not held_out:
        raise ValueError("Both train and eval conversations are required")
    one_step = [
        replay_task(record, boundary)
        for record in train
        for boundary in pick_boundaries(record, max_boundaries)
    ]
    if not one_step:
        raise ValueError("No training replies violate the style policy")
    return TaskSets(
        one_step=one_step,
        online=[conversation_task(record) for record in train],
        evaluation=[conversation_task(record) for record in held_out],
    )
