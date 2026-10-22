"""Native Gaussian position control for walking, archived rate control otherwise."""

import math
from dataclasses import dataclass
from typing import cast

import torch

from .guided_transitions import (
    GuidedOmniCommand,
    GuidedTransitionAction,
    GuidedTransitionActionCfg,
)
from .task_fsm import TaskState


@dataclass(kw_only=True)
class NativeGuidedTransitionActionCfg(GuidedTransitionActionCfg):
    native_biped_scale: float = .25

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.native_biped_scale) or self.native_biped_scale <= 0:
            raise ValueError('native_biped_scale must be finite and positive')

    def build(self, env):
        return NativeGuidedTransitionAction(self, env)


class NativeGuidedTransitionAction(GuidedTransitionAction):
    """Preserve native walking actions without bounding their Gaussian samples.

    The other three tasks retain the guided action's integration, mirror and
    physical straightening rules. Its persistent target also remains available
    when entering a transition from a native walking target.
    """

    def process_actions(self, actions):
        command = cast(GuidedOmniCommand, self._env.command_manager.get_term('transform'))
        walking = (command.current_task == TaskState.BIPED_MOTION)[:, None]
        # A walking Gaussian sample is not a bounded rate command. Do not feed
        # it into the inherited rate-control path, even temporarily.
        super().process_actions(torch.where(walking, 0., actions))
        cfg = cast(NativeGuidedTransitionActionCfg, self.rate_cfg)
        nominal = self._entity.data.default_joint_pos[:, self._target_ids]
        desired = actions * cfg.native_biped_scale + nominal
        self._raw_actions.copy_(torch.where(walking, actions, self._raw_actions))
        self._processed_actions.copy_(torch.where(walking, desired, self.commanded_target))
