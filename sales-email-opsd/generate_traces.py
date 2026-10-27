"""Optionally generate new fictional sales conversations through a model API.

The bundled traces need no regeneration. Use a new --data-dir for a new experiment.
Completed conversations are checkpointed; resume requires the same generation settings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from ac2.runtime import CompletionClient, Message, ModelConfiguration
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from sales_email_opsd.agent import SYSTEM_PROMPT
from sales_email_opsd.data import (
    ChatMessage,
    Conversation,
    Persona,
    content_hash,
    load_conversations,
    write_jsonl,
)
from sales_email_opsd.personas import build_personas
from sales_email_opsd.text import message_text

GUIDELINES = """Simulate the sales rep in this fictional persona. Write one short chat message.
First request an outbound email using the product, prospect, trigger, goal, and tone.
On later turns, give one piece of feedback from the iteration moves. Do not invent facts
or comment on punctuation or typography. When satisfied, set done to true.
Return only JSON with exactly two keys: "message" (string) and "done" (boolean).

Persona:
"""
GENERATOR_VERSION = "api-conversations-v2"


class UserReply(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: str
    done: bool


async def complete(client: CompletionClient, messages: list[Message]) -> str:
    result = await asyncio.wait_for(client.complete(items=messages), timeout=180.0)
    text = "\n".join(
        message_text(item) for item in result.items if isinstance(item, Message) and item.role == "assistant"
    ).strip()
    if not text:
        raise ValueError("The model returned no visible response")
    return text


async def generate_conversation(
    conversation_id: str,
    split: str,
    persona: Persona,
    model: str,
) -> Conversation:
    client = CompletionClient(ModelConfiguration(model=model))
    messages: list[ChatMessage] = []
    for turn in range(persona.n_user_turns):
        history = [
            Message(role="assistant" if m.role == "user" else "user", content=m.content) for m in messages
        ] or [Message(role="user", content="Write your opening request.")]
        raw = await complete(
            client, [Message(role="system", content=GUIDELINES + persona.render()), *history]
        )
        reply = UserReply.model_validate_json(raw)
        if reply.done and turn > 0:
            break
        messages.append(ChatMessage(role="user", content=reply.message))
        response = await complete(
            client,
            [
                Message(role="system", content=SYSTEM_PROMPT),
                *(Message(role=m.role, content=m.content) for m in messages),
            ],
        )
        messages.append(ChatMessage(role="assistant", content=response))
    return Conversation(conversation_id=conversation_id, split=split, persona=persona, messages=messages)


def generation_plan(n_train: int, n_eval: int, seed: int, model: str) -> list[tuple[str, str, Persona]]:
    if n_train < 1 or n_eval < 1:
        raise ValueError("Both split sizes must be positive")
    jobs = []
    for split, size in (("train", n_train), ("eval", n_eval)):
        rng = random.Random(f"{seed}:{split}")
        for index, persona in enumerate(build_personas(size, rng)):
            fingerprint = json.dumps(
                [
                    GENERATOR_VERSION,
                    model,
                    seed,
                    split,
                    index,
                    SYSTEM_PROMPT,
                    GUIDELINES,
                    persona.model_dump(),
                ],
                sort_keys=True,
            )
            jobs.append((f"{split}_{content_hash(fingerprint)[:20]}", split, persona))
    return jobs


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-eval", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-4.1-2025-04-14")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    load_dotenv()
    jobs = generation_plan(args.n_train, args.n_eval, args.seed, args.model)
    path = args.data_dir / "traces.jsonl"
    records = {r.conversation_id: r for r in load_conversations(path)} if path.exists() else {}
    if set(records) - {job[0] for job in jobs}:
        parser.error("Existing traces use different settings; use a new --data-dir or the original settings")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "generation.json").write_text(
        json.dumps(
            {
                "generator_version": GENERATOR_VERSION,
                "model": args.model,
                "seed": args.seed,
                "n_train": args.n_train,
                "n_eval": args.n_eval,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    semaphore = asyncio.Semaphore(args.workers)

    async def generate(job: tuple[str, str, Persona]) -> None:
        async with semaphore:
            record = await generate_conversation(*job, args.model)
            records[record.conversation_id] = record
            write_jsonl(path, [records[key] for key in sorted(records)])
            print(f"Saved {len(records)}/{len(jobs)} conversations", flush=True)

    await asyncio.gather(*(generate(job) for job in jobs if job[0] not in records))
    load_conversations(path)


if __name__ == "__main__":
    asyncio.run(main())
