"""蛇形六关节绝对目标角控制：默认姿态加缩放动作，目标角上限与动作尺度独立。"""

import math
from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class DirectJointPositionActionCfg(JointPositionActionCfg):
    target_limit: float = math.pi

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.target_limit) or self.target_limit <= 0:
            raise ValueError('关节目标角上限必须为正有限值')
        if not isinstance(self.scale, (float, int)) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('蛇形动作缩放必须为正有限标量')

    def build(self, env):
        return DirectJointPositionAction(self, env)


class DirectJointPositionAction(JointPositionAction):
    def __init__(self, cfg: DirectJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self.position_cfg = cfg
        self.reset()

    @property
    def commanded_target(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions.copy_(torch.nan_to_num(actions).clamp(-1., 1.))
        # 对齐G1：默认姿态加缩放动作；不依赖上一步目标或控制步长。
        target = self._raw_actions * self._scale + self._offset
        self._processed_actions.copy_(self._clamp_target(target))

    def _clamp_target(self, target, env_ids=slice(None)):
        limits = self._entity.data.joint_pos_limits[env_ids][:, self._target_ids]
        limit = self.position_cfg.target_limit
        return target.clamp(
            min=limits[..., 0].clamp(min=-limit),
            max=limits[..., 1].clamp(max=limit),
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        baseline = self._offset[ids] if isinstance(self._offset, torch.Tensor) else self._offset
        target = torch.zeros_like(self._processed_actions[ids]) + baseline
        self._processed_actions[ids] = self._clamp_target(target, ids)
