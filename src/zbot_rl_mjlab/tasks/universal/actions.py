"""全能策略按阶段约束关节目标，保留双足和变形所需范围。"""

import math
from dataclasses import dataclass
from typing import cast

import torch

from ..transformation.actions import JointTargetRateAction, JointTargetRateActionCfg
from .commands import UniversalCommand


@dataclass(kw_only=True)
class SnakeCrawlJointTargetRateActionCfg(JointTargetRateActionCfg):
    snake_target_limit: float = math.pi / 3
    biped_residual_scale: float | None = None
    transition_end_joint_radius: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.snake_target_limit) or self.snake_target_limit <= 0:
            raise ValueError('snake_target_limit must be finite and positive')
        if self.biped_residual_scale is not None and (
            not math.isfinite(self.biped_residual_scale) or self.biped_residual_scale <= 0
        ):
            raise ValueError('biped_residual_scale must be finite and positive')
        if self.transition_end_joint_radius is not None and (
            not math.isfinite(self.transition_end_joint_radius) or self.transition_end_joint_radius <= 0
        ):
            raise ValueError('transition_end_joint_radius must be finite and positive')

    def build(self, env):
        return SnakeCrawlJointTargetRateAction(self, env)


class SnakeCrawlJointTargetRateAction(JointTargetRateAction):
    def process_actions(self, actions):
        scale = cast(SnakeCrawlJointTargetRateActionCfg, self.rate_cfg).biped_residual_scale
        previous = self.commanded_target.clone() if scale is not None else None
        super().process_actions(actions)
        if scale is None:
            return
        command = cast(UniversalCommand, self._env.command_manager.get_term('transform'))
        walking = command.standing_mode & ~command.transition_mask
        nominal = self._entity.data.default_joint_pos[:, self._target_ids]
        desired = self._clamp_target(nominal + scale * self.raw_action)
        # Preserve target continuity when entering walking from a transformation.
        assert previous is not None
        max_delta = self.rate_cfg.max_target_speed * self._env.step_dt
        target = previous + (desired - previous).clamp(-max_delta, max_delta)
        self._processed_actions.copy_(torch.where(walking[:, None], target, self.commanded_target))

    def _target_limits(self):
        limits = super()._target_limits()
        if getattr(self, '_resetting', False):
            return limits
        command = cast(UniversalCommand, self._env.command_manager.get_term('transform'))
        radius = cast(SnakeCrawlJointTargetRateActionCfg, self.rate_cfg).transition_end_joint_radius
        if radius is not None:
            if not hasattr(self, '_transition_target_bounds'):
                nominal = self._entity.data.default_joint_pos[:, self._target_ids]
                span = nominal.new_tensor([radius if name in ('joint1', 'joint6') else math.inf for name in self._target_names])
                self._transition_target_bounds = torch.stack((nominal - span, nominal + span), -1)
            bounds = self._transition_target_bounds
            transition_limits = torch.stack((
                torch.maximum(limits[..., 0], bounds[..., 0]),
                torch.minimum(limits[..., 1], bounds[..., 1]),
            ), -1)
            limits = torch.where(command.transition_mask[:, None, None], transition_limits, limits)
        crawling = ~command.standing_mode & ~command.transition_mask
        limit = cast(SnakeCrawlJointTargetRateActionCfg, self.rate_cfg).snake_target_limit
        bounded = torch.stack((limits[..., 0].clamp(min=-limit), limits[..., 1].clamp(max=limit)), -1)
        return torch.where(crawling[:, None, None], bounded, limits)

    def reset(self, env_ids=None):
        # 环境先重置动作、后重置指令；不能用上一回合的蛇形模式裁掉新双足初态。
        self._resetting = True
        try:
            super().reset(env_ids)
        finally:
            self._resetting = False
