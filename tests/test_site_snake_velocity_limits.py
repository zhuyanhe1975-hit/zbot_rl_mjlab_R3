"""Continuous snake commands are independent of inherited deployment speed stages."""
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.tasks.snake.site_omni import SiteSnakeCommandCfg


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('limits', [(.1, .1, .2), (.03, .02, .1), (.3, .15, .5)])
def test_continuous_snake_accepts_limits_below_unused_deployment_stages(training, limits):
    env = SimpleNamespace(num_envs=128, device='cpu', step_dt=.02, common_step_counter=0)
    cfg = SiteSnakeCommandCfg(training=training, velocity_limits=limits)
    term = cfg.build(env)
    term.command[:, 0] = -1
    term._sample_velocity(torch.arange(env.num_envs))
    assert (term.requested_velocity.abs() <= torch.tensor(limits)).all()
    term.set_velocity(*[2*value for value in limits])
    torch.testing.assert_close(term.requested_velocity, torch.tensor(limits).expand(env.num_envs, -1))


@pytest.mark.parametrize('limits', [(0., .1, .2), (-.1, .1, .2), (.1, float('nan'), .2)])
def test_continuous_snake_still_rejects_invalid_velocity_bounds(limits):
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0)
    with pytest.raises(ValueError, match='速度范围'):
        SiteSnakeCommandCfg(velocity_limits=limits).build(env)
