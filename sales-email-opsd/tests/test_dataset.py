from pathlib import Path

import pytest
from ac2.runtime import HintSelection, Task
from ac2.runtime.base.hinter import validate_hint_selection

from build_opsd_dataset import build
from sales_email_opsd.agent import SYSTEM_PROMPT
from sales_email_opsd.data import Conversation, load_conversations, write_jsonl
from sales_email_opsd.dataset import build_tasks, make_hint, pick_boundaries, replay_task
from upload_dataset import load_bundle

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def hint_selection(task: Task) -> HintSelection:
    [selection] = task.grader_params["opsd_hint_selections"]
    return HintSelection.model_validate(selection)


def test_replay_stops_at_exact_boundary_with_system_prompt(conversation: Conversation) -> None:
    boundary = pick_boundaries(conversation, 1)[0]
    task = replay_task(conversation, boundary)
    selection = hint_selection(task)
    validate_hint_selection(task.input, selection)
    assert [m.content for m in task.input] == [
        SYSTEM_PROMPT,
        *[m.content for m in conversation.messages[: boundary + 1]],
    ]
    assert selection.rollout_prefix_len == len(task.input) - 1
    assert task.input[selection.rollout_prefix_len].role == "assistant"


def test_hint_is_appended_to_the_rep_request(conversation: Conversation) -> None:
    task = replay_task(conversation, pick_boundaries(conversation, 1)[0])
    mutation = hint_selection(task).mutation
    assert mutation.action == "append_to_previous_item"
    assert task.input[mutation.item_boundary - 1].role == "user"


def test_hint_is_teacher_only(conversation: Conversation) -> None:
    task = replay_task(conversation, pick_boundaries(conversation, 1)[0])
    assert all(hint_selection(task).hint not in m.content for m in task.input)


@pytest.mark.parametrize("reply", ["a—b", "a - b", "a – b", "a--b"])
def test_hints_cover_every_punctuation_variant(reply: str) -> None:
    assert reply in make_hint(reply)


def test_compliant_reply_cannot_be_a_replay_target() -> None:
    with pytest.raises(ValueError, match="offending"):
        make_hint("20-minute call")


def test_boundary_limit_is_positive(conversation: Conversation) -> None:
    with pytest.raises(ValueError, match="positive"):
        pick_boundaries(conversation, 0)


def test_corpus_splits_are_disjoint_and_eval_has_no_hints() -> None:
    tasks = build_tasks(load_conversations(DATA_DIR / "traces.jsonl"))
    train_ids = {t.tags["conversation_id"] for t in tasks.online}
    assert train_ids.isdisjoint(t.tags["conversation_id"] for t in tasks.evaluation)
    assert all(t.grader_params == {} for t in tasks.evaluation)


def test_no_explicit_style_policy_in_bundled_user_turns() -> None:
    records = load_conversations(DATA_DIR / "traces.jsonl")
    assert not any(
        term in m.content.casefold()
        for r in records
        for m in r.messages
        if m.role == "user"
        for term in ("em dash", "em-dash", "punctuation", "typography")
    )


def test_invalid_turn_sequence_is_rejected(conversation: Conversation) -> None:
    row = conversation.model_dump()
    row["messages"][1]["role"] = "user"
    with pytest.raises(ValueError, match="must have role assistant"):
        Conversation.model_validate(row)


def test_duplicate_ids_are_rejected(tmp_path: Path, conversation: Conversation) -> None:
    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [conversation, conversation])
    with pytest.raises(ValueError, match="Duplicate"):
        load_conversations(path)


def test_same_opening_cannot_cross_splits(tmp_path: Path, conversation: Conversation) -> None:
    other = conversation.model_copy(update={"conversation_id": "held_out", "split": "eval"})
    path = tmp_path / "traces.jsonl"
    write_jsonl(path, [conversation, other])
    with pytest.raises(ValueError, match="both train and eval"):
        load_conversations(path)


def test_rebuild_is_byte_for_byte_reproducible(tmp_path: Path) -> None:
    (tmp_path / "traces.jsonl").write_bytes((DATA_DIR / "traces.jsonl").read_bytes())
    build(tmp_path)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    build(tmp_path)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_bundle_detects_modified_output(tmp_path: Path) -> None:
    for path in DATA_DIR.glob("*.json*"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    with (tmp_path / "tasks_eval.jsonl").open("a") as file:
        file.write("{}\n")
    with pytest.raises(ValueError, match="manifest"):
        load_bundle(tmp_path)


def test_bundled_files_match_current_builder() -> None:
    bundle = load_bundle(DATA_DIR)
    assert [len(tasks) for tasks in bundle.values()] == [347, 120, 30]
