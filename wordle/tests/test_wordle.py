import itertools
import json
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from ac2.runtime import FunctionCall, Message
from ac2.runtime.completions.client import CompletionClient

from wordle.agent import WordleAgent
from wordle.audit import audit
from wordle.dataset import EVAL_DATASET, TRAIN_DATASET, build_datasets
from wordle.environment import WordleEnvironment
from wordle.grader import WordleGrader
from wordle.train import CONFIG, monitor
from wordle.user import WordleUser
from wordle.words import GREEN, WHITE, YELLOW, dictionary, normalize_word, score_guess


def call(guess, index=0):
    return FunctionCall(call_id=str(index), name="check_answer", arguments=json.dumps({"guess": guess}))


class ScoringTests(unittest.TestCase):
    def test_examples(self):
        for answer, guess, expected in [
            ("APPLE", "APPLE", "🟩🟩🟩🟩🟩"),
            ("APPLE", "GUESS", "⬜⬜🟨⬜⬜"),
            ("APPLE", "ALLEY", "🟩🟨⬜🟨⬜"),
            ("APPLE", "PAPAL", "🟨🟨🟩⬜🟨"),
            ("CLOSE", "LEAVE", "🟨⬜⬜⬜🟩"),
            ("THOSE", "GEESE", "⬜⬜⬜🟩🟩"),
            ("ABBEY", "KEEPS", "⬜🟨⬜⬜⬜"),
            ("BANAL", "LLAMA", "🟨⬜🟨⬜🟨"),
            ("DREAD", "ADDED", "🟨🟨⬜🟨🟩"),
        ]:
            with self.subTest(answer=answer, guess=guess):
                self.assertEqual(score_guess(answer, guess), expected)

    def test_exhaustive_letter_accounting(self):
        # 59,049 pairs cover every repeat/position pattern over three letters.
        words = ["".join(letters) for letters in itertools.product("ABC", repeat=5)]
        for answer in words:
            for guess in words:
                tiles = score_guess(answer, guess)
                self.assertEqual([t == GREEN for t in tiles],
                                 [a == g for a, g in zip(answer, guess)])
                matched = Counter(g for g, t in zip(guess, tiles) if t != WHITE)
                self.assertEqual(matched, Counter(answer) & Counter(guess))
                for letter in "ABC":
                    non_green = [t for g, t in zip(guess, tiles) if g == letter and t != GREEN]
                    self.assertEqual(non_green, sorted(non_green, key=lambda t: t != YELLOW))

    def test_normalization(self):
        self.assertEqual(normalize_word(" \tapPle\n"), "APPLE")
        for value in ["", "four", "longer", "a pple", "caféx", "ＡＰＰＬＥ", "ſassy",
                      "aßes", "12345", "apple!", None, 12345, ["apple"]]:
            with self.subTest(value=value):
                self.assertIsNone(normalize_word(value))

    def test_dictionary_and_split(self):
        self.assertEqual(len(dictionary()), 8636)
        self.assertTrue(all(normalize_word(w) == w for w in dictionary()))
        datasets = build_datasets()
        self.assertEqual(len(datasets[TRAIN_DATASET]), 2048)
        self.assertEqual(len(datasets[EVAL_DATASET]), 256)
        train = {t.env_params["base_word"] for t in datasets[TRAIN_DATASET]}
        evaluation = {t.env_params["base_word"] for t in datasets[EVAL_DATASET]}
        self.assertFalse(train & evaluation)
        self.assertEqual(len(train | evaluation), 2304)
        self.assertEqual(datasets, build_datasets())
        prompts = {t.input[0].content for tasks in datasets.values() for t in tasks}
        self.assertEqual(len(prompts), 1)
        self.assertTrue(all(not t.grader_params for tasks in datasets.values() for t in tasks))


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = WordleEnvironment()
        await self.env.setup({"base_word": "apple"})

    async def reward(self):
        return (await WordleGrader().grade(None, [], self.env)).score

    async def test_only_tool_and_hidden_answer(self):
        self.assertEqual([t.name for t in self.env.tools], ["check_answer"])
        self.assertEqual(await self.env.inspect(["base_word", "_base_word"]), {})

    async def test_all_six_reward_levels(self):
        for k in range(1, 7):
            await self.env.setup({"base_word": "APPLE"})
            for _ in range(k - 1):
                _, done = await self.env.step([call("crane")])
                self.assertFalse(done)
                self.assertEqual(await self.reward(), 0)
            outputs, done = await self.env.step([call(" \tApple\n")])
            self.assertTrue(done)
            self.assertEqual(outputs[0].output, "🟩🟩🟩🟩🟩")
            self.assertEqual(self.env.guess_count, k)
            self.assertEqual(await self.reward(), [1, .9, .8, .7, .6, .5][k - 1])

    async def test_exhaustion_and_late_win(self):
        for i in range(6):
            _, done = await self.env.step([call("crane", i)])
            self.assertEqual(done, i == 5)
        self.assertEqual(await self.reward(), 0)
        self.assertEqual(await self.env.check_answer("apple"), "invalid guess")
        self.assertEqual(await self.env.step([call("apple")]), ([], True))
        self.assertEqual(self.env.guess_count, 6)
        self.assertFalse(self.env.solved)

    async def test_invalid_guesses_are_bounded(self):
        for guess in ["zzzzz", "four", "12345", "ſassy", "a pple", None]:
            outputs, _ = await self.env.step([call(guess)])
            self.assertEqual(outputs[0].output, "invalid guess")
        self.assertEqual(self.env.terminated_reason, "exhausted")
        self.assertEqual(await self.reward(), 0)

    async def test_malformed_calls_are_bounded(self):
        for name, arguments in [("check_answer", "not JSON"), ("check_answer", "{}"),
                                ("check_answer", '[]'), ("check_answer", '{"answer":"apple"}'),
                                ("teardown", "{}"), ("finish", "{}")]:
            outputs, _ = await self.env.step([FunctionCall(call_id="x", name=name, arguments=arguments)])
            self.assertEqual(outputs[0].output, "invalid guess")
        self.assertEqual(self.env.guess_count, 6)
        self.assertTrue(self.env.is_terminated)

    async def test_batched_calls_stop_at_win_or_limit(self):
        outputs, done = await self.env.step([call("crane"), call("apple", 1), call("crane", 2)])
        self.assertTrue(done)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(await self.reward(), .9)
        await self.env.setup({"base_word": "APPLE"})
        outputs, done = await self.env.step([call("crane", i) for i in range(10)] + [call("apple")])
        self.assertTrue(done)
        self.assertEqual(len(outputs), 6)
        self.assertEqual(await self.reward(), 0)

    async def test_win_is_immutable_and_setup_resets(self):
        await self.env.check_answer("apple")
        await self.env.check_answer("crane")
        self.assertEqual(await self.reward(), 1)
        await self.env.setup({"base_word": "CRANE"})
        self.assertEqual(self.env.guess_count, 0)
        self.assertFalse(self.env.is_terminated)
        self.assertFalse(self.env.solved)
        self.assertEqual(await self.env.check_answer("crane"), "🟩🟩🟩🟩🟩")

    async def test_text_is_not_a_winning_submission(self):
        _, done = await self.env.step([Message(role="assistant", content="APPLE")])
        self.assertTrue(done)
        self.assertFalse(self.env.is_terminated)
        self.assertEqual(self.env.guess_count, 0)
        self.assertEqual(await self.reward(), 0)

    async def test_invalid_task_fails(self):
        for params in [{}, {"base_word": "zzzzz"}, {"base_word": "bad"}]:
            with self.assertRaises(ValueError):
                await self.env.setup(params)


class DeadlineTests(unittest.TestCase):
    def test_expired_deadline_stops_remote_run(self):
        client = Mock()
        self.assertEqual(monitor(client, "train_test", -1), "deadline")
        client.train.stop.assert_called_once_with("train_test", stop_evals=True)
        client.train.get.assert_not_called()

    def test_completion_does_not_stop(self):
        client = Mock()
        client.train.get.return_value = SimpleNamespace(status="completed")
        self.assertEqual(monitor(client, "train_test", float("inf")), "completed")
        client.train.stop.assert_not_called()

    def test_polling_failure_and_interrupt_stop_run(self):
        for error in [RuntimeError("connection failed"), KeyboardInterrupt()]:
            client = Mock()
            client.train.get.side_effect = error
            with self.assertRaises(type(error)):
                monitor(client, "train_test", float("inf"))
            client.train.stop.assert_called_once_with("train_test", stop_evals=True)

    def test_running_job_reaches_deadline(self):
        client = Mock()
        client.train.get.return_value = SimpleNamespace(status="running")
        with patch("wordle.train.time.monotonic", side_effect=[0, 1, 61]), \
             patch("wordle.train.time.sleep"):
            self.assertEqual(monitor(client, "train_test", 60), "deadline")
            client.train.stop.assert_called_once_with("train_test", stop_evals=True)


class SamplingTests(unittest.IsolatedAsyncioTestCase):
    async def test_eval_and_training_preserve_thinking_at_their_api_boundaries(self):
        # Training uses its own client; agent kwargs alone do not affect it.
        payload = CONFIG.to_payload(train_tasks=[], eval_tasks=[],
                                    training_agent_names=["WordleAgent"])
        self.assertEqual(payload["apply_chat_template_kwargs"], {"enable_thinking": True})
        client = CompletionClient(WordleAgent.model_configuration)
        client._adapter = Mock(complete=AsyncMock())
        await client.complete([Message(role="user", content="Make a guess.")])
        sent = client._adapter.complete.call_args.kwargs
        self.assertEqual(sent["extra_body"]["chat_template_kwargs"],
                         payload["apply_chat_template_kwargs"])


class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_answer_can_resume_without_resetting_episode_or_guess_count(self):
        from ac2.runtime import Task
        from ac2.runtime.base.orchestrator import canonical_orchestrator
        from ac2.runtime.completions.types import Usage
        from ac2.runtime.orchestration.eval import run_eval

        class Reply:
            def __init__(self, items):
                self.result = SimpleNamespace(items=items, usage=Usage())

            async def __aiter__(self):
                if False:
                    yield

        env, agent = WordleEnvironment(), WordleAgent()
        replies = iter([
            Reply([Message(role="assistant", content="The answer is APPLE.")]),
            Reply([call("apple")]),
        ])
        orchestrator = canonical_orchestrator(agent, env)
        episode_ids = []

        async def complete(**kwargs):
            episode_ids.append(orchestrator.trace[0].id)
            return next(replies)

        agent.get_completion = AsyncMock(side_effect=complete)
        task = Task(input=[Message(role="user", content="Make a guess.")],
                    env_params={"base_word": "apple"})
        results = await run_eval(lambda: orchestrator, WordleGrader(),
                                 tasks_by_dataset={"test": [task]}, num_samples=1,
                                 max_parallel=1, on_error="raise", user_factory=WordleUser)
        self.assertEqual(results[0].grades[0].score, 1)
        self.assertEqual(env.guess_count, 1)
        self.assertEqual(len(set(episode_ids)), 1)
        self.assertEqual(agent.get_completion.call_count, 2)
        second_input = agent.get_completion.call_args.kwargs["items"]
        self.assertIn("written answer does not count", second_input[-1].content)

    async def test_reminders_are_bounded_and_reset_per_task(self):
        from ac2.runtime import Task

        task = Task(input=[Message(role="user", content="Make a guess.")])
        env, user = WordleEnvironment(), WordleUser()
        await env.setup({"base_word": "apple"})
        await user.setup(task)
        self.assertEqual(await user.respond(env, []), task.input)
        for _ in range(3):
            self.assertIsNotNone(await user.respond(env, []))
        self.assertIsNone(await user.respond(env, []))
        self.assertEqual(env.terminated_reason, "abandoned")
        self.assertEqual(env.guess_count, 0)
        await env.setup({"base_word": "apple"})
        await user.setup(task)
        self.assertEqual(await user.respond(env, []), task.input)
        await env.check_answer("apple")
        self.assertIsNone(await user.respond(env, []))


class AuditTests(unittest.TestCase):
    @staticmethod
    def trace(reason, guesses, solved=False, error=""):
        return {"scores": [{"grader_name": "WordleGrader", "score": 0,
                            "artifacts": json.dumps({"terminated_reason": reason,
                                                     "guesses": guesses, "solved": solved})}],
                "spans": [{"error_message": error}]}

    def test_completed_loss_passes_but_partial_games_do_not(self):
        complete = [self.trace("exhausted", 6), self.trace("correct", 2, True)]
        self.assertTrue(audit(complete)["passed"])
        result = audit(complete + [self.trace("", 3, error="SGLang returned finish_reason=length"),
                                   self.trace("abandoned", 4), {"scores": []}])
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["token_limit_hits"], 1)
        self.assertEqual(result["completion_rate"], .4)
        self.assertFalse(result["passed"])

    def test_empty_export_and_inconsistent_terminal_state_fail(self):
        self.assertFalse(audit([])["passed"])
        self.assertFalse(audit([self.trace("exhausted", 2)])["passed"])
        self.assertFalse(audit([self.trace("correct", 0, True)])["passed"])


if __name__ == "__main__":
    unittest.main()
