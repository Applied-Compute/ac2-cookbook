import base64
import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from ac2.runtime import Environment, Episode, Message
from PIL import Image
from pydantic import ValidationError

from byoh_geo3k_vlm.grader import Geo3KGrader, answer_score
from byoh_geo3k_vlm.tasks import GeometryProblem, row_to_task


def test_task_preserves_diagram_and_keeps_answer_out_of_prompt() -> None:
    diagram = Image.new("RGB", (2, 2), color="red")
    task = row_to_task(GeometryProblem(problem="<image>Find x.", images=[diagram], answer="3"))
    message = task.input[0]
    assert isinstance(message, Message)
    assert isinstance(message.content, list)
    assert message.content[0] == {"type": "text", "text": "Find x."}
    url = message.content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    restored = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert restored.tobytes() == diagram.tobytes()
    assert task.grader_params == {"answer": "3"}


def test_task_ids_are_stable_and_include_expected_answer() -> None:
    row = GeometryProblem(problem="Find x.", images=[Image.new("RGB", (2, 2))], answer="3")
    assert row_to_task(row).id == row_to_task(row).id
    assert row_to_task(row).id != row_to_task(row.model_copy(update={"answer": "4"})).id


def test_dataset_row_requires_an_image() -> None:
    with pytest.raises(ValidationError):
        GeometryProblem(problem="Find x.", images=[], answer="3")


@pytest.mark.parametrize(
    ("response", "expected", "score"),
    [
        (r"\boxed{\frac{1}{2}}", "0.5", 1),
        (r"\boxed{2\sqrt{6}}", r"\sqrt{24}", 1),
        (r"\boxed{4}", "3", 0),
        ("", "3", 0),
    ],
)
def test_math_equivalence_in_worker_thread(response: str, expected: str, score: int) -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(answer_score, response, expected).result(timeout=10) == score


@pytest.mark.asyncio
async def test_grader_uses_final_assistant_answer() -> None:
    episode = Episode("custom_harness")
    episode.add(
        [
            Message(role="assistant", content=r"\boxed{4}"),
            Message(role="assistant", content=r"\boxed{3}"),
            Message(role="user", content=r"I thought it was \boxed{5}."),
        ]
    )
    result = await Geo3KGrader()._grade({"answer": "3"}, [episode], Environment())
    assert result.score == 1
