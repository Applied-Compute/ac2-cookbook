"""Interactive local session against the rock-paper-scissors agent."""

from __future__ import annotations

import asyncio
import os

from ac2.runtime import Message
from ac2.sdk import Client
from dotenv import load_dotenv

from rock_paper_scissors import RPSAgent, RPSEnvironment

DEFAULT_PROMPT = "Play rock-paper-scissors against the mystery opponent. Begin:"


def print_session_event(event) -> None:
    if event.type in {"text_delta", "thinking_delta", "function_call_delta"}:
        print(event.content, end="", flush=True)
    elif event.type == "function_call_start":
        print(f"\n[calling {event.name or 'tool'}]", flush=True)
    elif event.type == "function_call_output":
        print(f"\n\033[38;2;240;74;0m[{event.name or 'tool'} returned]\033[0m {event.content}", flush=True)
    elif event.type == "turn_complete":
        print(flush=True)
    elif event.type == "error":
        print(f"\n[error]\n{event.content}", flush=True)


async def main(*, start: bool = False, deployment_id: str | None = None) -> None:
    client = Client(project="rock-paper-scissors")
    if deployment_id:
        session = client.session.from_remote(deployment_id)
        starting_prompt = None
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set; export it or put it in .env")
        session = client.session.from_local(agent=RPSAgent(), env=RPSEnvironment())
        starting_prompt = DEFAULT_PROMPT if start else None

    async with session:
        if starting_prompt:
            async for event in session.run([Message(role="user", content=starting_prompt)]):
                print_session_event(event)
        while True:
            try:
                text = await asyncio.to_thread(input, "\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            async for event in session.run([Message(role="user", content=text)]):
                print_session_event(event)


if __name__ == "__main__":
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(description="Interactive rock-paper-scissors session")
    parser.add_argument("--start", action="store_true", help="Send the default game prompt first")
    parser.add_argument("--deployment-id", default=None, help="Connect to a remote deployment")
    args = parser.parse_args()
    asyncio.run(main(start=args.start, deployment_id=args.deployment_id))
