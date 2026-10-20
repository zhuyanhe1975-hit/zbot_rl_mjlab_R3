"""Reward-group checkpoint isolation without allocating a simulator."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from zbot_rl_mjlab import grouped_teacher
from zbot_rl_mjlab.grouped_teacher import GroupedTeacherRunner, grouped_teacher_ppo_cfg
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunner, omni_teacher_ppo_cfg
from zbot_rl_mjlab.tasks.universal.reward_groups import GROUP_NAMES


@pytest.fixture
def runner_setup(monkeypatch):
    from zbot_rl_mjlab.tasks.universal import mdp

    term = SimpleNamespace(tracking_ema=torch.zeros(2))
    physical = SimpleNamespace(common_step_counter=7)
    env = SimpleNamespace(
        unwrapped=physical, num_actions=6,
        get_observations=lambda: {"actor": torch.zeros(2, 56)},
    )
    capture = Mock()
    monkeypatch.setattr(grouped_teacher, "install_reward_group_capture", capture)
    monkeypatch.setattr(mdp, "command_term", lambda env: term)

    def base_init(self, env, cfg, log_dir, device):
        capture.assert_called_once_with(physical)
        self.env, self.cfg = env, cfg
        self.alg = SimpleNamespace(update=Mock(return_value={}), actor=object())
        self.logger = SimpleNamespace(writer=None)
        self.current_learning_iteration = 0

    monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", base_init)
    runner = GroupedTeacherRunner(cast(RslRlVecEnvWrapper, env), asdict(grouped_teacher_ppo_cfg()))
    return runner, term


def save_checkpoint(path, runner, **changes):
    saved = {"infos": {
        "privileged_omni_teacher": runner._schema() | {"completed_ppo_iterations": 12} | changes,
        "universal_course": [.2, .6],
    }}
    torch.save(saved, path)
    return saved


def test_grouped_config_changes_only_runner_algorithm_and_critic_class():
    baseline = asdict(omni_teacher_ppo_cfg())
    cfg = asdict(grouped_teacher_ppo_cfg())
    assert cfg["class_name"] == "zbot_rl_mjlab.grouped_teacher:GroupedTeacherRunner"
    assert cfg["algorithm"]["class_name"] == "zbot_rl_mjlab.grouped_ppo:GroupedPPO"
    assert cfg["critic"]["class_name"] == "zbot_rl_mjlab.grouped_ppo:IndependentRewardCritics"
    cfg["class_name"] = baseline["class_name"]
    cfg["algorithm"]["class_name"] = baseline["algorithm"]["class_name"]
    cfg["critic"]["class_name"] = baseline["critic"]["class_name"]
    assert cfg == baseline
    assert asdict(omni_teacher_ppo_cfg()) == baseline


def test_schema_and_playback_select_scalar_encoding(runner_setup, tmp_path):
    runner, _ = runner_setup
    schema = runner._schema()
    assert schema["role"] == "privileged_omni_reward_group_teacher"
    assert schema["reward_groups"] == GROUP_NAMES
    path = tmp_path / "grouped.pt"
    save_checkpoint(path, runner)
    command = SimpleNamespace(task_observation_encoding="one_hot")
    GroupedTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={"transform": command}), path)
    assert command.task_observation_encoding == "scalar"
    with pytest.raises(ValueError):
        OmniTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={"transform": command}), path)


@pytest.mark.parametrize("change", [
    {"role": "privileged_omni_teacher"}, {"training_algorithm": "PPO"},
    {"reward_groups": tuple(reversed(GROUP_NAMES))}, {"reward_groups": list(GROUP_NAMES)},
    {"schema_version": True}, {"actor_obs_dim": 59}, {"critic_obs_dim": 59}, {"action_dim": 3},
])
@pytest.mark.parametrize("load_cfg", [None, {"actor": True}])
def test_mismatched_checkpoint_rejected_before_any_weight_mutation(
    monkeypatch, runner_setup, tmp_path, change, load_cfg,
):
    runner, term = runner_setup
    path = tmp_path / "invalid.pt"
    save_checkpoint(path, runner, **change)
    load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    with pytest.raises(ValueError, match="role/schema/algorithm"):
        runner.load(path, load_cfg)
    load.assert_not_called()
    torch.testing.assert_close(term.tracking_ema, torch.zeros(2))
    assert runner.current_learning_iteration == runner.completed_ppo_iterations == 0
    command = SimpleNamespace(task_observation_encoding="untouched")
    with pytest.raises(ValueError):
        GroupedTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={"transform": command}), path)
    assert command.task_observation_encoding == "untouched"


@pytest.mark.parametrize("count", [-1, True])
def test_invalid_resume_progress_reuses_parent_guard(monkeypatch, runner_setup, tmp_path, count):
    runner, _ = runner_setup
    path = tmp_path / "invalid_progress.pt"
    save_checkpoint(path, runner, completed_ppo_iterations=count)
    load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    with pytest.raises(ValueError, match="iteration count"):
        runner.load(path)
    load.assert_not_called()


def test_full_resume_restores_parent_course_and_completed_updates(monkeypatch, runner_setup, tmp_path):
    runner, term = runner_setup
    path = tmp_path / "valid.pt"
    saved = save_checkpoint(path, runner)
    load = Mock(return_value=saved["infos"])
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    runner.load(path)
    load.assert_called_once()
    assert runner.current_learning_iteration == runner.completed_ppo_iterations == 12
    torch.testing.assert_close(term.tracking_ema, torch.tensor([.2, .6]))


def test_legacy_observation_dimension_rejected(runner_setup):
    runner, _ = runner_setup
    runner.env.get_observations = lambda: {"actor": torch.zeros(2, 59)}
    with pytest.raises(ValueError, match="56-dimensional"):
        runner._schema()
