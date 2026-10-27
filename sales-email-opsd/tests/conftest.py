from pathlib import Path

import pytest

from sales_email_opsd.data import Conversation, load_conversations

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def conversation() -> Conversation:
    return load_conversations(DATA_DIR / "traces.jsonl")[0]
