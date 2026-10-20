"""头倾终止扣分是真实的一次性-5，不随控制步长或其他复位变化。"""

import math

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.utils.lab_api.math import quat_mul
from test_snake_task import state_env


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_head_tilt_event_charges_once_and_is_independent_of_control_dt(dt):
    env = state_env()
    env.step_dt = dt
    env.max_episode_length_s = 30.
    manager = TerminationManager({'head_tilt': env.cfg.terminations['head_tilt']}, env)
    env.termination_manager = manager
    rewards = RewardManager({'head_tilt_reset_cost': env.cfg.rewards['head_tilt_reset_cost']}, env)
    manager.compute()
    torch.testing.assert_close(rewards.compute(dt), torch.zeros(2))
    quaternions = env.scene['robot'].data.body_link_quat_w
    initial = quaternions[0, 0].clone()
    tilt = torch.tensor([math.cos(math.pi / 6), math.sin(math.pi / 6), 0., 0.])
    quaternions[0, 0] = quat_mul(tilt, initial)
    torch.testing.assert_close(manager.compute(), torch.tensor([True, False]))
    torch.testing.assert_close(rewards.compute(dt), torch.tensor([-5., 0.]))
    # 原生回合日志按最大回合秒数归一化，实际事件费用仍为-5。
    logged = rewards.reset(torch.tensor([0]))
    assert logged['Episode_Reward/head_tilt_reset_cost'].item() == pytest.approx(-5. / 30)
    manager.reset(torch.tensor([0]))
    quaternions[0, 0] = initial
    assert not manager.compute().any()
    torch.testing.assert_close(rewards.compute(dt), torch.zeros(2))


@pytest.mark.parametrize('reason', ['unsafe', 'self_collision', 'time_out'])
def test_other_reset_reasons_do_not_charge_head_tilt_fee(reason):
    env = state_env()
    manager = TerminationManager({
        'head_tilt': env.cfg.terminations['head_tilt'],
        reason: TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool),
                                   time_out=reason == 'time_out'),
    }, env)
    env.termination_manager = manager
    assert manager.compute().all()
    rewards = RewardManager({'head_tilt_reset_cost': env.cfg.rewards['head_tilt_reset_cost']}, env)
    torch.testing.assert_close(rewards.compute(env.step_dt), torch.zeros(2))
