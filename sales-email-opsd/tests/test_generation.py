from generate_traces import generation_plan


def test_growing_training_split_does_not_move_eval_personas() -> None:
    small = generation_plan(2, 2, 42, "model-a")
    large = generation_plan(4, 2, 42, "model-a")
    assert small[2:] == large[4:]


def test_model_changes_invalidate_resume_ids() -> None:
    first = {row[0] for row in generation_plan(2, 2, 42, "model-a")}
    second = {row[0] for row in generation_plan(2, 2, 42, "model-b")}
    assert first.isdisjoint(second)
