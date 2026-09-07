import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from wordle.agent import WordleAgent, WordleQwen36Agent
from wordle.train import main, monitor
from wordle.train_qwen36 import CONFIG, validate_allocation


class AllocationTests(unittest.TestCase):
    def test_four_hour_budget_dry_run_and_upper_bound(self):
        for args in [[], ["--max-minutes", "235"], ["--max-minutes", "20"]]:
            with self.subTest(args=args), patch("sys.argv", ["train_qwen36", "--dry-run", *args]), \
                    patch("wordle.train.Client") as client, contextlib.redirect_stdout(io.StringIO()):
                main(CONFIG, default_max_minutes=235)
                client.assert_not_called()
        for minutes in ["0", "-1", "236", "nan", "inf"]:
            with self.subTest(minutes=minutes), \
                    patch("sys.argv", ["train_qwen36", "--dry-run", "--max-minutes", minutes]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(CONFIG, default_max_minutes=235)
            self.assertEqual(error.exception.code, 2)

    def test_accepts_four_plus_four_and_rejects_multiplied_allocation(self):
        flags = {"actor-num-nodes": 1, "actor-num-gpus-per-node": 4,
                 "rollout-num-gpus": 4, "rollout-num-gpus-per-engine": 2,
                 "tensor-model-parallel-size": 4, "context-parallel-size": 1,
                 "expert-model-parallel-size": 4, "expert-tensor-parallel-size": 1}
        command = ["bash", "-c", "python train.py " + " ".join(f"--{k}={v}" for k, v in flags.items())]
        resources = SimpleNamespace(num_nodes=1, gpus="8", gpu_type="b200")
        validate_allocation(resources, command)
        with self.assertRaises(RuntimeError):
            validate_allocation(SimpleNamespace(num_nodes=4, gpus="8", gpu_type="b200"), command)
        with self.assertRaises(RuntimeError):
            validate_allocation(resources, [*command[:-1], command[-1].replace("--rollout-num-gpus=4", "--rollout-num-gpus=8")])

    def test_watchdog_retries_failed_stop_with_attached_evals(self):
        client = Mock()
        client.train.stop.side_effect = [ConnectionError(), None]
        with patch("wordle.train.time.sleep"):
            self.assertEqual(monitor(client, "train_test", -1), "deadline")
        self.assertEqual(client.train.stop.call_count, 2)
        client.train.stop.assert_called_with("train_test", stop_evals=True)

    def test_model_specific_tool_instructions(self):
        self.assertIn('<function=check_answer>', WordleQwen36Agent.system_prompt)
        self.assertNotIn('"arguments"', WordleQwen36Agent.system_prompt)
        self.assertIn('"arguments"', WordleAgent.system_prompt)
