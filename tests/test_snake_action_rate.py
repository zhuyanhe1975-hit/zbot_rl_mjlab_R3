"""原生G1动作变化成本抑制高频目标抖动，保留平滑周期运动。"""

import math

import pytest
import torch
from mjlab.envs import mdp as common_mdp
from mjlab.managers.reward_manager import RewardManager
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


@pytest.mark.parametrize('play', [False, True])
def test_train_and_play_use_g1_action_rate_cost(play):
    term = snake_env_cfg(play=play).rewards['action_rate']
    assert term.func is common_mdp.action_rate_l2
    assert term.weight == -.0


def test_action_rate_uses_raw_action_difference_and_constant_actions_are_free():
    env = state_env()
    env.action_manager.prev_action[:] = .5
    env.action_manager.action[:] = .5
    torch.testing.assert_close(common_mdp.action_rate_l2(env), torch.zeros(2))
    env.action_manager.action[0] = -.5
    torch.testing.assert_close(common_mdp.action_rate_l2(env), torch.tensor([6., 0.]))
    # 目标角/实际关节变化不参与这个原始action成本。
    env.action_manager.get_term('joint_pos').commanded_target[:] = math.pi
    env.scene['robot'].data.joint_pos[:] = -1.
    torch.testing.assert_close(common_mdp.action_rate_l2(env), torch.tensor([6., 0.]))


def test_high_frequency_alternation_costs_more_than_smooth_snake_wave():
    t = torch.arange(500) * .02
    phase = torch.arange(6) * math.pi / 3
    wave = torch.sin(2 * math.pi * 1.2 * t[:, None] + phase)
    jitter = (2 * (torch.arange(500) % 2) - 1).float()[:, None].expand(-1, 6)
    env = state_env()
    costs = []
    for trajectory in (wave, jitter):
        values = []
        for current, previous in zip(trajectory[1:], trajectory[:-1], strict=True):
            env.action_manager.action[:] = current
            env.action_manager.prev_action[:] = previous
            values.append(common_mdp.action_rate_l2(env)[0])
        costs.append(torch.stack(values).mean())
    assert costs[1] == 24.
    assert costs[1] > 100 * costs[0]


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_penalty_is_applied_by_reward_manager_with_native_dt_scaling(dt):
    env = state_env()
    env.action_manager.action[:] = 1.
    env.action_manager.prev_action[:] = -1.
    env.cfg.rewards['action_rate'].weight = -.01
    manager = RewardManager({'action_rate': env.cfg.rewards['action_rate']}, env)
    torch.testing.assert_close(manager.compute(dt), torch.full((2,), -.01 * 24 * dt))
