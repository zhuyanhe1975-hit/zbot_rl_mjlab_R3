"""前进速度线性奖励在0.2m/s截断，倒退不截断。"""

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp


@pytest.mark.parametrize('speed', [-1., -.2, -.01, 0., .01, .1, .2, .3, .4, 1., 1000.])
def test_capped_speed_matches_definition_and_stays_finite(speed):
    env = state_env()
    env.scene['robot'].data.body_com_lin_vel_w[..., 0] = speed
    expected = min(speed, .2)
    torch.testing.assert_close(mdp.head_forward_speed(env), torch.full((2,), expected))
    torch.testing.assert_close(mdp.head_forward_velocity(env), torch.full((2,), speed))
    assert torch.isfinite(mdp.head_forward_speed(env)).all()


def test_reward_is_linear_below_cap_and_constant_above_it():
    env = state_env()
    speeds = torch.linspace(0., 1., 101)
    values = []
    for speed in speeds:
        env.scene['robot'].data.body_com_lin_vel_w[..., 0] = speed
        values.append(mdp.head_forward_speed(env)[0])
    values = torch.stack(values)
    assert (values[1:21] > values[:20]).all()
    torch.testing.assert_close(values[:21], speeds[:21])
    torch.testing.assert_close(values[20:], torch.full_like(values[20:], .2))
    assert values[20] == pytest.approx(.2)
    assert values[0] == 0


def test_reward_manager_keeps_cap_scale_and_applies_dt():
    env = state_env()
    env.scene['robot'].data.body_com_lin_vel_w[..., 0] = .2
    manager = RewardManager({'head_forward_speed': env.cfg.rewards['head_forward_speed']}, env)
    torch.testing.assert_close(manager.compute(.02), torch.full((2,), 15 * .2 * .02))


def test_cap_can_be_configured_without_changing_projection():
    env = state_env()
    env.cfg.forward_speed_cap = .1
    env.scene['robot'].data.body_com_lin_vel_w[..., 0] = .1
    torch.testing.assert_close(mdp.head_forward_speed(env), torch.full((2,), .1))


@pytest.mark.parametrize('cap', [0., -1., float('nan'), float('inf')])
def test_invalid_cap_speed_is_rejected(cap):
    env = state_env()
    env.cfg.forward_speed_cap = cap
    with pytest.raises(ValueError):
        mdp.head_forward_speed(env)
