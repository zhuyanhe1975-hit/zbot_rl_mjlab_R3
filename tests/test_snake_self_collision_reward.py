"""自碰撞终止的原生一次性费用不依赖控制步长。"""

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from test_snake_task import state_env


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_self_collision_charges_five_points_once_per_reset_independent_of_dt(dt):
    env = state_env()
    env.step_dt = dt
    env.max_episode_length_s = 30.
    term = TerminationManager({'self_collision': env.cfg.terminations['self_collision']}, env)
    env.termination_manager = term
    rewards = RewardManager({'self_collision_reset_cost': env.cfg.rewards['self_collision_reset_cost']}, env)
    term.compute()
    torch.testing.assert_close(rewards.compute(dt), torch.zeros(2))
    contact = env.scene['snake_self_contact'].data
    contact.found[0] = 1
    torch.testing.assert_close(term.compute(), torch.tensor([True, False]))
    torch.testing.assert_close(rewards.compute(dt), torch.tensor([-5., 0.]))
    rewards.reset(torch.tensor([0]))
    term.reset(torch.tensor([0]))
    contact.found.zero_()
    assert not term.compute().any()
    torch.testing.assert_close(rewards.compute(dt), torch.zeros(2))
