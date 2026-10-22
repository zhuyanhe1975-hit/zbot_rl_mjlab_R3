"""Paired forward-only ablation of instantaneous lateral/yaw tracking costs."""
from dataclasses import dataclass
from typing import cast

from zbot_rl_mjlab.tasks.snake.site_omni import (
    SiteSnakeCommand,
    SiteSnakeCommandCfg,
    _velocity_tracking_reward,
)

from ..universal import mdp


@dataclass(kw_only=True)
class ForwardSnakeCommandCfg(SiteSnakeCommandCfg):
    play_velocity: tuple[float, float, float] = (.1, 0., 0.)

    def build(self, env):
        return ForwardSnakeCommand(self, env)


class ForwardSnakeCommand(SiteSnakeCommand):
    def _sample_velocity(self, ids):
        ids = ids[~self.manual_control[ids]]
        cfg = cast(ForwardSnakeCommandCfg, self.cfg)
        self.requested_velocity[ids] = self.requested_velocity.new_tensor(cfg.play_velocity)


def tracking_component(env, axis):
    return _velocity_tracking_reward(env)[:, axis]


def target_hard_limit_fraction(env):
    action = env.action_manager.get_term('joint_pos')
    limit = env.cfg.actions['joint_pos'].snake_target_limit
    return (action.commanded_target.abs() >= limit - 1e-3).float().mean(-1)


def mean_abs_joint_velocity(env):
    return env.scene['robot'].data.joint_vel.abs().mean(-1)




def velocity_component(env, axis):
    return mdp.tracking_velocity(env)[:, axis]
