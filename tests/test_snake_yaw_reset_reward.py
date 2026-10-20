"""yaw失败费用是一次性事件，和头倾、自碰撞一样避免廉价退出。"""

import math

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.utils.lab_api.math import quat_mul
from test_snake_task import state_env


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_yaw_failure_charges_five_points_once_and_reset_restores_alive(dt):
    env = state_env()
    env.step_dt = dt
    env.max_episode_length_s = 30.
    env.termination_manager = TerminationManager({'head_yaw': env.cfg.terminations['head_yaw']}, env)
    rewards = RewardManager({name: env.cfg.rewards[name] for name in ['head_yaw_reset_cost', 'alive']}, env)
    env.termination_manager.compute()
    torch.testing.assert_close(rewards.compute(dt), torch.full((2,), .02 * dt))
    data = env.scene['robot'].data
    initial = data.body_link_quat_w[0, 0].clone()
    angle = math.radians(61) / 2
    data.body_link_quat_w[0, 0] = quat_mul(torch.tensor([math.cos(angle), 0., 0., math.sin(angle)]), initial)
    env.termination_manager.compute()
    torch.testing.assert_close(rewards.compute(dt), torch.tensor([-5., .02 * dt]))
    logged = rewards.reset(torch.tensor([0]))
    assert logged['Episode_Reward/head_yaw_reset_cost'].item() == pytest.approx(-5 / 30)
    env.termination_manager.reset(torch.tensor([0]))
    data.body_link_quat_w[0, 0] = initial
    env.termination_manager.compute()
    torch.testing.assert_close(rewards.compute(dt), torch.full((2,), .02 * dt))


def test_other_failure_does_not_charge_yaw_reset_fee():
    env = state_env()
    env.termination_manager = TerminationManager({
        'head_yaw': env.cfg.terminations['head_yaw'],
        'unsafe': TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool)),
    }, env)
    env.termination_manager.compute()
    rewards = RewardManager({'head_yaw_reset_cost': env.cfg.rewards['head_yaw_reset_cost']}, env)
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.zeros(2))
