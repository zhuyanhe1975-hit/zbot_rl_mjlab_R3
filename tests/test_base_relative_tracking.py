"""Experiment20 restores bounded relative scores while retaining contribution feedback."""
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from test_site_snake_runner import setup as setup  # noqa: PLC0414
from test_site_snake_task import reward_env

from zbot_rl_mjlab.adaptive_site_snake import VELOCITY_REWARDS
from zbot_rl_mjlab.base_snake_teacher import BaseSnakeAdaptiveTeacherRunner
from zbot_rl_mjlab.experiments.site_snake import base_snake_env_cfg
from zbot_rl_mjlab.tasks.snake.site_omni import (
    angular_tracking,
    forward_tracking,
    lateral_tracking,
    relative_velocity_scores,
)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('command', [-.2, -.01, .01, .2])
def test_relative_ratio_points_and_minus_two_clamp(monkeypatch, axis, command):
    ratios = torch.tensor([-1., 0., .25, .5, .75, 1., 1.5, 2., 10.])
    target, actual = torch.zeros(9, 3), torch.zeros(9, 3)
    target[:, axis] = command
    actual[:, axis] = command*ratios
    env = reward_env(monkeypatch, actual.tolist(), target.tolist())
    reward = (forward_tracking, lateral_tracking, angular_tracking)[axis]
    torch.testing.assert_close(reward(env), torch.tensor([-2., -1., -.5, 0., .5, 1., 0., -1., -2.]), atol=1e-6, rtol=1e-5)


def test_base_factory_uses_relative_rewards_and_no_std_params():
    cfg = base_snake_env_cfg()
    for name, function in zip(VELOCITY_REWARDS, (forward_tracking, lateral_tracking, angular_tracking), strict=True):
        assert cfg.rewards[name].func is function
        assert cfg.rewards[name].params == {}


def test_zero_command_stop_is_zero_and_drift_negative(monkeypatch):
    env = reward_env(monkeypatch, [[0., 0., 0.], [.1, 0., 0.]], [[0., 0., 0.]]*2)
    torch.testing.assert_close(forward_tracking(env), torch.tensor([0., -.5]))


def test_adaptive_feedback_keeps_negative_scores_and_disables_positive_ratio():
    cfg = base_snake_env_cfg()
    runner: Any = object.__new__(BaseSnakeAdaptiveTeacherRunner)
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg))
    runner.velocity_balancer = BaseSnakeAdaptiveTeacherRunner.velocity_balancer_cls(3.)
    raw = torch.zeros(3*256, 61)
    expected = [1., -.5, -2.]
    for axis, score in enumerate(expected):
        ids = slice(axis*256, (axis+1)*256)
        raw[ids, 40+axis] = .1
        raw[ids, 49+axis] = .1*(1.+(1.-score)/2.)
    measured = runner._measure_velocity(raw, torch.tensor([.2, .2, .5]))
    torch.testing.assert_close(measured[0], torch.tensor(expected), atol=1e-6, rtol=1e-5)
    metrics = runner._advance_velocity_weights(measured)
    assert metrics['velocity_score_vy'] == pytest.approx(-.5, abs=1e-6)
    assert metrics['velocity_score_ema_wz'] == pytest.approx(-2., abs=1e-6)
    assert metrics['velocity_reward_yaw_mean'] < 0
    assert metrics['velocity_reward_ratio_valid'] == 0.
    assert runner.velocity_balancer.reward_mean_ema[2] < 0


def test_checkpoint_keeps_feedback_state_and_rejects_exponential_recipe(tmp_path):
    cfg = base_snake_env_cfg(play=True)
    saved = {'infos': {'privileged_omni_teacher': BaseSnakeAdaptiveTeacherRunner._metadata_schema() | {
        'velocity_limits': (.1, .2, .3), 'joint_target_limit': torch.pi/2,
        'snake_target_limit': torch.pi/2, 'max_target_speed': 2*torch.pi},
        'adaptive_velocity_weights': BaseSnakeAdaptiveTeacherRunner.velocity_balancer_cls(3.).state_dict()}}
    path = tmp_path/'linear.pt'
    torch.save(saved, path)
    BaseSnakeAdaptiveTeacherRunner.prepare_playback_env_cfg(cfg, path)
    assert cfg.rewards['track_forward_velocity'].params == {}
    saved['infos']['privileged_omni_teacher']['reward_recipe'] = 'base_snake_exponential_reward_ratio_v4'
    with pytest.raises(ValueError, match='compatible'):
        BaseSnakeAdaptiveTeacherRunner._validate_checkpoint(saved)


def test_training_wrapper_uses_signed_relative_scores_and_real_reward_capture(setup, tmp_path):
    env, train_cfg, alg, *_ = setup
    cfg = base_snake_env_cfg()
    env.unwrapped.cfg, env.device = cfg, 'cpu'
    terms = deepcopy(cfg.rewards)
    manager = SimpleNamespace(get_term_cfg=lambda name: terms[name], active_terms=list(terms),
                             compute=lambda dt: torch.zeros(768), _step_reward=torch.zeros(768, len(terms)))
    env.unwrapped.reward_manager = manager
    train_cfg['validation_interval_iterations'] = train_cfg['robust_validation_interval'] = 0
    raw = torch.zeros(1, 768, 61)
    raw[..., 39] = 2.
    expected = [1., -.5, -2.]
    for axis, score in enumerate(expected):
        ids = slice(axis*256, (axis+1)*256)
        raw[0, ids, 40+axis] = .1
        raw[0, ids, 49+axis] = .1*(1.+(1.-score)/2.)
    alg.storage = SimpleNamespace(observations={'actor': raw})
    runner = BaseSnakeAdaptiveTeacherRunner(env, train_cfg, log_dir=str(tmp_path))
    scores = relative_velocity_scores(raw[0, :, 40:43], raw[0, :, 49:52], raw.new_tensor([.2, .2, .5]))
    for axis, name in enumerate(VELOCITY_REWARDS):
        manager._step_reward[:, manager.active_terms.index(name)] = scores[:, axis]*runner.velocity_balancer.weights[axis]
    manager.compute(.02)
    metrics = runner.alg.update()
    for axis, score in zip(('vx', 'vy', 'wz'), expected, strict=True):
        assert metrics[f'velocity_score_{axis}'] == pytest.approx(score, abs=1e-6)
    assert metrics['velocity_reward_ratio_target'] == .25
