"""小额生存奖励按时间缩放，失败帧不给，正常超时不算失败。"""

import pytest
import torch
from mjlab.envs import mdp as common_mdp
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from test_snake_task import state_env

from zbot_rl_mjlab.experiments.snake import snake_env_cfg


@pytest.mark.parametrize('play', [False, True])
def test_alive_uses_native_failure_mask_and_small_weight(play):
    term = snake_env_cfg(play=play).rewards['alive']
    assert term.func is common_mdp.is_alive
    assert term.weight == .02


@pytest.mark.parametrize('dt', [.01, .02, .04])
@pytest.mark.parametrize('reason', ['head_yaw', 'body_yaw', 'head_tilt', 'self_collision', 'unsafe', 'time_out'])
def test_alive_excludes_failure_frames_and_preserves_timeout_and_other_env(dt, reason):
    env = state_env()
    env.failed = torch.tensor([True, False])
    term = TerminationTermCfg(func=lambda env: env.failed, time_out=reason == 'time_out')
    env.termination_manager = TerminationManager({reason: term}, env)
    env.termination_manager.compute()
    rewards = RewardManager({'alive': env.cfg.rewards['alive']}, env)
    expected = torch.ones(2) if reason == 'time_out' else torch.tensor([0., 1.])
    torch.testing.assert_close(common_mdp.is_alive(env), expected)
    torch.testing.assert_close(rewards.compute(dt), expected * .02 * dt)
    # 只重置失败环境后，该环境的生存判据恢复；不留下上回合的终止标记。
    env.termination_manager.reset(torch.tensor([0]))
    env.failed.zero_()
    env.termination_manager.compute()
    torch.testing.assert_close(rewards.compute(dt), torch.full((2,), .02 * dt))


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_survival_reward_rate_does_not_depend_on_control_dt(dt):
    env = state_env()
    manager = RewardManager({'alive': env.cfg.rewards['alive']}, env)
    total = sum(manager.compute(dt) for _ in range(round(1 / dt)))
    torch.testing.assert_close(total, torch.full((2,), .02))
