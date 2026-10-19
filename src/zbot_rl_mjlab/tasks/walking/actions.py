"""用于面向部署训练的随机控制延迟动作。"""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """关节位置目标延迟 ``lag`` 个控制步后到达执行器。

    每次重置时，逐环境从 0..max_lag 均匀采样 ``lag``，让策略体验真实控制环路
    （感知、计算、总线）的延迟，同时无法直接读取延迟值。``last_action`` 观测
    和动作变化率惩罚仍使用策略自身的原始动作。
    """

    max_lag: int = 2

    def build(self, env):
        return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg=cfg, env=env)
        if cfg.max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        # history[0] 为最新指令；history[k] 为 k 步之前的指令。
        self._history = torch.zeros(
            cfg.max_lag + 1, self.num_envs, self.action_dim, device=self.device
        )
        self._lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rows = torch.arange(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        count = self.num_envs if isinstance(ids, slice) else len(ids)
        self._history[:, ids] = 0.0
        self._lag[ids] = torch.randint(
            0, self.cfg.max_lag + 1, (count,), device=self._lag.device
        )

    def process_actions(self, actions):
        self._history = torch.roll(self._history, shifts=1, dims=0)
        self._history[0] = actions
        super().process_actions(self._history[self._lag, self._rows])
