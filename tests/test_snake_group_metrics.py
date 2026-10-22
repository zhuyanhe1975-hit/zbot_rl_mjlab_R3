"""指令分组按请求互斥，统计按每环境曝光量归一化。"""
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg

from zbot_rl_mjlab.experiments.snake import snake_env_cfg
from zbot_rl_mjlab.tasks.snake import mdp


def group_env(monkeypatch):
    request = torch.tensor([[.04, 0., 0.], [0., 0., .2], [0., 0., 0.], [.03, .04, .2]])
    term = SimpleNamespace(command=request.clone(), requested_velocity=request,
                           cfg=SimpleNamespace(velocity_limits=(.25, .25, 1.5)),
                           translation_stage=1, current_translation_cap=.1)
    actual = torch.tensor([[0., 0., 0.], [0., 0., .2], [0., 0., 0.], [.03, .04, .1]])
    monkeypatch.setattr(mdp, 'command_term', lambda env: term)
    monkeypatch.setattr(mdp, 'planar_velocity', lambda env: actual)
    # Test configured reward execution, not a duplicate of the objective formula.
    rewards = {
        'translation_tracking': SimpleNamespace(
            func=lambda env, std: torch.exp(-((actual[:, :2] - term.command[:, :2]) / std).square().sum(-1)),
            weight=5., params={'std': .05}),
        'yaw_tracking': SimpleNamespace(func=lambda env: torch.tensor([1., 1., 1., .9]), weight=3., params={}),
    }
    env = SimpleNamespace(num_envs=4, device='cpu', cfg=SimpleNamespace(rewards=rewards))
    return env, term, actual


def test_requested_command_groups_partition_all_environments(monkeypatch):
    env, term, _ = group_env(monkeypatch)
    masks = torch.stack([mdp.command_group_mask(env, group) for group in mdp.COMMAND_GROUPS])
    torch.testing.assert_close(masks, torch.eye(4, dtype=torch.bool))
    term.command.zero_()  # Smoothing cannot relabel newly requested moving commands as stop.
    assert mdp.command_group_mask(env, 'translation')[0]
    assert mdp.command_group_mask(env, 'mixed')[3]


def test_stationary_success_is_separate_from_failed_translation(monkeypatch):
    env, _, _ = group_env(monkeypatch)
    translation = mdp.command_group_metric(env, 'translation', 'linear_reward')
    stop = mdp.command_group_metric(env, 'stop', 'linear_reward')
    assert translation[0].item() == pytest.approx(5 * torch.exp(torch.tensor(-.64)).item())
    torch.testing.assert_close(stop, torch.tensor([0., 0., 5., 0.]))
    torch.testing.assert_close(mdp.command_group_metric(env, 'mixed', 'yaw_reward'), torch.tensor([0., 0., 0., 2.7]))
    assert mdp.command_group_metric(env, 'translation', 'linear_error')[0] == pytest.approx(.04)
    assert mdp.command_group_metric(env, 'mixed', 'target_speed')[3] == pytest.approx(.05)
    assert mdp.command_group_metric(env, 'mixed', 'actual_speed')[3] == pytest.approx(.05)
    assert mdp.command_group_metric(env, 'mixed', 'yaw_error')[3] == pytest.approx(.1)


def test_metrics_manager_keeps_matching_episode_exposure_denominators(monkeypatch):
    env, term, actual = group_env(monkeypatch)
    cfg = {metric: MetricsTermCfg(func=mdp.command_group_metric,
                                params={'group': 'translation', 'metric': metric})
           for metric in ('fraction', 'actual_speed')}
    manager = MetricsManager(cfg, cast(ManagerBasedRlEnv, env))
    actual[0, 0] = .02
    manager.compute()
    term.requested_velocity[0].zero_()
    manager.compute()
    extras = manager.reset(torch.tensor([0, 1]))
    fraction = extras['Episode_Metrics/fraction']
    contribution = extras['Episode_Metrics/actual_speed']
    assert fraction.item() == pytest.approx(.25)
    assert (contribution / fraction).item() == pytest.approx(.02)
    empty = manager.reset(torch.tensor([2, 3]))
    assert empty['Episode_Metrics/fraction'] == 0
    assert empty['Episode_Metrics/actual_speed'] == 0


def test_current_task_omits_group_metrics_but_keeps_legacy_helpers(monkeypatch):
    env, _, _ = group_env(monkeypatch)
    cfg = snake_env_cfg()
    assert len(cfg.metrics) == 19
    assert not any(name.startswith(('group_', 'translation_curriculum_')) for name in cfg.metrics)
    for axis, name in enumerate(('x', 'y', 'z')):
        assert cfg.metrics[f'world_velocity_{name}'].func is mdp.world_velocity_component
        assert cfg.metrics[f'world_velocity_{name}'].params == {'axis': axis}
    torch.testing.assert_close(mdp.translation_curriculum_metric(env, 'cap'), torch.full((4,), .1))
    torch.testing.assert_close(mdp.translation_curriculum_metric(env, 'stage'), torch.ones(4))
