import unittest
from PIL import Image
from geo3k_vlm.tasks import row_to_task
from geo3k_vlm.grader import answer_score
from geo3k_vlm.train import config


class ExampleTests(unittest.TestCase):
    def test_image_task(self):
        row = {"problem": "<image>Find x.", "images": [Image.new("RGB", (2, 2))], "answer": "3"}
        task = row_to_task(row)
        self.assertEqual(task.input[0].content[0]["text"], "Find x.")
        self.assertTrue(task.input[0].content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(task.grader_params, {"answer": "3"})
        self.assertEqual(task.id, row_to_task(row).id)
        with self.assertRaises(ValueError):
            row_to_task({**row, "images": []})

    def test_math_grading(self):
        self.assertEqual(answer_score(r"Therefore \boxed{\frac{1}{2}}", "0.5"), 1)
        self.assertEqual(answer_score(r"\boxed{2\sqrt{6}}", r"\sqrt{24}"), 1)
        self.assertEqual(answer_score(r"\boxed{4}", "3"), 0)
        self.assertEqual(answer_score("", "3"), 0)

    def test_math_grading_in_worker_thread(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(answer_score, r"\boxed{\frac{1}{2}}", "0.5")
            self.assertEqual(result.result(timeout=10), 1)

    def test_catalog_config(self):
        value = config("example-cluster")
        self.assertTrue(value.model.endswith("-VL"))
        self.assertEqual(value.extra_train_args, {})
        self.assertEqual(value.ac2_agent, "Geo3KAgent")


class GraderTests(unittest.IsolatedAsyncioTestCase):
    async def test_last_assistant_answer(self):
        from types import SimpleNamespace
        from ac2.runtime import Message
        from geo3k_vlm.grader import Geo3KGrader
        episode = SimpleNamespace(get_items=lambda: [
            Message(role="user", content="Find x."),
            Message(role="assistant", content=r"\boxed{3}"),
        ])
        result = await Geo3KGrader()._grade({"answer": "3"}, [episode], None)
        self.assertEqual(result.score, 1)
