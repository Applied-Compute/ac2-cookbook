import unittest

from harness_server import answer_score


class AnswerScoreTests(unittest.TestCase):
    def test_answer_score(self):
        self.assertEqual(answer_score(r"Therefore \boxed{\frac{1}{2}}", "0.5"), 1)
        self.assertEqual(answer_score(r"\boxed{2\sqrt{6}}", r"\sqrt{24}"), 1)
        self.assertEqual(answer_score(r"The answer is $\frac{1}{2}$.", "0.5"), 1)
        self.assertEqual(answer_score(r"\boxed{4}", "3"), 0)
        self.assertEqual(answer_score("", "3"), 0)


if __name__ == "__main__":
    unittest.main()
