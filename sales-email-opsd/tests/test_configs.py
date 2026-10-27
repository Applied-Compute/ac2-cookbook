import pytest
from ac2.sdk import ModelConfig

from sales_email_opsd.agent import BASE_MODEL
from sales_email_opsd.eval import eval_config
from sales_email_opsd.train_opsd import training_config
from sales_email_opsd.user import ReplaySalesEmailUser, SalesEmailUser


def test_baseline_uses_the_training_model() -> None:
    assert eval_config().model == training_config().model == BASE_MODEL


def test_checkpoint_eval_selects_only_checkpoint() -> None:
    config = eval_config(checkpoint="ckpt_example")
    assert config.model == ModelConfig(checkpoint_id="ckpt_example")


def test_model_selections_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="Choose one"):
        eval_config(model=BASE_MODEL, api_model="gpt-4.1")


def test_paired_eval_replays_fixed_user_turns() -> None:
    assert isinstance(eval_config().user, ReplaySalesEmailUser)


def test_provider_eval_uses_requested_model() -> None:
    config = eval_config(api_model="my-project/endpoint")
    assert config.model is None
    assert config.agent.model_configuration.model == "my-project/endpoint"


@pytest.mark.parametrize("mode, engines", [("offline", 0), ("one_step", 4), ("online", 4)])
def test_mode_engine_requirements(mode: str, engines: int) -> None:
    assert training_config(mode).n_inference_replicas == engines


@pytest.mark.parametrize("mode", ["offline", "one_step"])
def test_replay_modes_do_not_include_rollout_components(mode: str) -> None:
    config = training_config(mode)
    assert (config.ac2_agent, config.ac2_env, config.ac2_user, config.ac2_grader) == (None, None, None, None)


def test_smoke_is_one_step_with_ten_tasks() -> None:
    config = training_config("offline", smoke=True)
    assert (config.num_train_steps, config.ac2_train_dataset[0].num_tasks) == (1, 10)


def test_online_mode_selects_loss_tokens_with_the_judge() -> None:
    assert training_config("online").judge_model == "gpt-5-mini"


@pytest.mark.parametrize("mode", ["offline", "one_step"])
def test_replay_modes_train_every_response_token(mode: str) -> None:
    assert training_config(mode).judge_model is None


def test_token_judge_can_be_disabled() -> None:
    assert training_config("online", token_judge=None).judge_model is None


def test_online_mode_simulates_the_rep_by_default() -> None:
    assert isinstance(training_config("online").ac2_user, SalesEmailUser)


def test_online_mode_can_replay_recorded_rep_turns() -> None:
    assert isinstance(training_config("online", replay_user=True).ac2_user, ReplaySalesEmailUser)


def test_replayed_users_require_online_mode() -> None:
    with pytest.raises(ValueError, match="online"):
        training_config("one_step", replay_user=True)


def test_online_mode_selects_hints_with_the_judge_hinter() -> None:
    hinter_path = training_config("online").extra_train_args["rmsd_hinter_path"]
    assert hinter_path == "sales_email_opsd.hinter.SalesEmailJudgeHinter"
