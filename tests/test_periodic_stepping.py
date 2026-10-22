"""原地踏步跟踪步频并抑制水平移动，行走实验继续奖励前进。"""

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.tasks.walking import mdp


@pytest.mark.parametrize('play', [False, True])
def test_periodic_stepping_has_no_forward_incentive(play):
    cfg = load_env_cfg('Mjlab-Zbot-6dof-Periodic-Stepping', play=play)
    assert 'forward_velocity' not in cfg.rewards
    assert cfg.rewards['planar_velocity'].func is mdp.planar_velocity_penalty
    assert cfg.rewards['planar_velocity'].weight < 0
    assert cfg.rewards['frequency_tracking'].weight > 0


@pytest.mark.parametrize('play', [False, True])
def test_walking_keeps_forward_reward_without_stepping_penalty(play):
    cfg = load_env_cfg('Mjlab-Zbot-6dof-Walking', play=play)
    assert cfg.rewards['forward_velocity'].func is mdp.forward_velocity_reward
    assert cfg.rewards['forward_velocity'].weight == 2.0
    assert 'planar_velocity' not in cfg.rewards


def test_planar_velocity_cost_is_symmetric_and_allows_vertical_motion(monkeypatch):
    monkeypatch.setattr(mdp, 'base_lin_vel', lambda env: torch.tensor([
        [0., 0., 1.], [.2, .3, 0.], [-.2, -.3, 1.],
    ]))
    torch.testing.assert_close(
        mdp.planar_velocity_penalty(None), torch.tensor([0., .13, .13])
    )
