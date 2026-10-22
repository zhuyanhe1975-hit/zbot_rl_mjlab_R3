"""Site vx/vy rewards must be independently weighted and independently scored."""
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab import archived_site_snake as task
from zbot_rl_mjlab.experiments.cards import load_card

TASK = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'


def test_axis_errors_are_independent(monkeypatch):
    target = torch.tensor([[.1, .1, 0.]] * 3)
    actual = torch.tensor([[.1, .1, 0.], [.1, -.1, 5.], [-.1, .1, -5.]])
    monkeypatch.setattr(task, 'command', lambda _: target)
    monkeypatch.setattr(task.mdp, 'tracking_velocity', lambda _: actual)
    env = SimpleNamespace()
    vx, vy = task.forward_tracking(env), task.lateral_tracking(env)
    torch.testing.assert_close(vx[:2], torch.ones(2))
    torch.testing.assert_close(vy[[0, 2]], torch.ones(2))
    assert 0. < vx[2] < 1. and 0. < vy[1] < 1.


def test_axis_bandwidth_parameters_can_be_tuned_independently(monkeypatch):
    target = torch.tensor([[.1, .1, 0.]])
    monkeypatch.setattr(task, 'command', lambda _: target)
    monkeypatch.setattr(task.mdp, 'tracking_velocity', lambda _: torch.zeros_like(target))
    env = SimpleNamespace()
    narrow = task.forward_tracking(env, std=.02)
    wide = task.forward_tracking(env, std=.2)
    assert wide.item() > narrow.item()
    torch.testing.assert_close(task.lateral_tracking(env), task.forward_tracking(env))


@pytest.mark.parametrize('play', [False, True])
def test_card_registers_separate_vx_vy_terms_with_configurable_weights(play):
    cfg = load_env_cfg(TASK, play=play)
    assert 'track_linear_velocity' not in cfg.rewards
    vx, vy = cfg.rewards['track_forward_velocity'], cfg.rewards['track_lateral_velocity']
    assert vx.func is task.forward_tracking and vy.func is task.lateral_tracking
    card = load_card('snake_site_omni_teacher')
    assert vx.weight == card['rewards']['track_forward_velocity']['weight']
    assert vy.weight == card['rewards']['track_lateral_velocity']['weight']
    assert vx.params == card['rewards']['track_forward_velocity']['params']
    assert vy.params == card['rewards']['track_lateral_velocity']['params']


def test_fixed_bandwidth_does_not_change_with_command_magnitude(monkeypatch):
    target = torch.tensor([[.05, 0., 0.], [.15, .1, 0.], [-.2, -.2, .5]])
    actual = target - torch.tensor([.1, 0., 0.])
    monkeypatch.setattr(task, 'command', lambda _: target)
    monkeypatch.setattr(task.mdp, 'tracking_velocity', lambda _: actual)
    reward = task.forward_tracking(SimpleNamespace(), std=.1)
    torch.testing.assert_close(reward, torch.full((3,), torch.exp(torch.tensor(-1.)).item()))


@pytest.mark.parametrize('std', [0., -.1, float('nan'), float('inf'), -float('inf')])
def test_invalid_fixed_bandwidth_rejected(std):
    for reward in (task.forward_tracking, task.lateral_tracking):
        with pytest.raises(ValueError, match='std'):
            reward(SimpleNamespace(), std=std)
