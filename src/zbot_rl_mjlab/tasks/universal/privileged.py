"""全技能教师的仿真真值输入；Actor与Critic共享一份特权状态。"""

import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

from ..common import constant
from . import mdp
from .observation_layout import (
    LEGACY_PRIVILEGED_OMNI_OBS_DIM,
    PRIVILEGED_OMNI_OBS_DIM,
)
from .omni import OmniCommand, OmniCommandCfg
from .task_fsm import Morphology, TaskStateMachine
from .teacher_sampling import sample_motion_commands
from .teacher_sampling import training_roles as assign_training_roles

__all__ = [
    'LEGACY_PRIVILEGED_OMNI_OBS_DIM', 'PRIVILEGED_OMNI_OBS_DIM',
    'PrivilegedOmniCommand', 'PrivilegedOmniCommandCfg', 'privileged_state', 'training_skill_complete',
]


@dataclass(kw_only=True)
class PrivilegedOmniCommandCfg(OmniCommandCfg):
    debug_vis_selected_only: bool = True
    task_observation_encoding: Literal['scalar', 'one_hot'] = 'one_hot'
    velocity_limits: tuple[float, float, float] = (.5, .2, .8)
    motion_axis_min_fraction: float = .4
    motion_stop_probability: float = 1 / 8
    training_task_fractions: tuple[float, float, float, float] = (.4, .4, .1, .1)
    randomize_play_modes: bool = True
    play_random_velocity: bool = False
    play_velocity: tuple[float, float, float] = (.05, 0., 0.)
    initialize_matched_motion: bool = True
    biped_gait_frequency_range: tuple[float, float] = (.25, 1.)
    snake_gait_frequency: float = .7
    standing_ready_height: float = .24
    transition_joint_tolerance: float = .15
    hold_s: float = 2.
    motion_startup_s: float = .1
    train_auto_recovery: bool = False
    play_auto_recovery: bool = False

    def build(self, env):
        return PrivilegedOmniCommand(self, env)


class PrivilegedOmniCommand(OmniCommand):
    """沿用全向目标采样，教师自行学习变形，不使用部署归直阶段。"""

    def _debug_vis_impl(self, visualizer):
        from ..snake.commands import SnakeVelocityCommand

        SnakeVelocityCommand._debug_vis_impl(cast(Any, self), visualizer)

    def __init__(self, cfg: PrivilegedOmniCommandCfg, env):
        if cfg.task_observation_encoding not in ('scalar', 'one_hot'):
            raise ValueError('task_observation_encoding must be scalar or one_hot')
        if not math.isfinite(cfg.motion_axis_min_fraction) or not 0 < cfg.motion_axis_min_fraction <= 1:
            raise ValueError('motion_axis_min_fraction must be in (0, 1]')
        if not math.isfinite(cfg.motion_stop_probability) or not 0 <= cfg.motion_stop_probability <= 1:
            raise ValueError('motion_stop_probability must be finite and in [0, 1]')
        x_only_probability = getattr(cfg, 'motion_x_only_probability', None)
        if x_only_probability is not None and (
            not math.isfinite(x_only_probability) or not 0 <= x_only_probability <= 1 - cfg.motion_stop_probability
        ):
            raise ValueError('motion_x_only_probability must be finite and in [0, 1-motion_stop_probability]')
        low, high = cfg.biped_gait_frequency_range
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
            raise ValueError('biped_gait_frequency_range must be finite and positive')
        if not math.isfinite(cfg.snake_gait_frequency) or cfg.snake_gait_frequency <= 0:
            raise ValueError('snake_gait_frequency must be finite and positive')
        goal_height = getattr(getattr(env, 'cfg', None), 'standing_base_height', .264)
        if not .22 < cfg.standing_ready_height <= goal_height:
            raise ValueError('standing_ready_height must exceed walking fall height and fit standing goal')
        if not math.isfinite(cfg.transition_joint_tolerance) or cfg.transition_joint_tolerance <= 0:
            raise ValueError('transition_joint_tolerance must be finite and positive')
        if not math.isfinite(cfg.motion_startup_s) or cfg.motion_startup_s < 0:
            raise ValueError('motion_startup_s must be finite and nonnegative')
        if cfg.train_auto_recovery or cfg.play_auto_recovery:
            raise ValueError('explicit task FSM requires automatic recovery disabled in train and play')
        super().__init__(cfg, env)
        self.teacher_cfg = cfg
        self._biped_gait_frequency = torch.full_like(self.gait_frequency, .5)
        self.canonical_motion_start = torch.full_like(self.start_step, -1)
        self._training_roles = assign_training_roles(
            self.num_envs, torch.arange(self.num_envs, device=self.device), cfg.training_task_fractions,
        )
        self.task_fsm = TaskStateMachine(self.num_envs, self.device)
        self.task_fsm.reset(torch.arange(self.num_envs, device=self.device), None, self.standing_mode)
        self._publish_task_state()

    @property
    def current_morphology(self):
        return self.task_fsm.current_morphology

    @property
    def target_morphology(self):
        return self.task_fsm.target_morphology

    @property
    def current_task(self):
        return self.task_fsm.task_state

    def _publish_task_state(self):
        with torch.inference_mode():
            self.phase.copy_(self.current_task)
            self.transition_mask.copy_(self.task_fsm.transition_mask)
            self.standing_achieved.logical_or_(self.current_morphology == Morphology.BIPED)

    def initialize_task_state(self, ids, initial_standing):
        self.task_fsm.reset(
            ids, initial_standing, self.standing_mode[ids],
            force_transition=not self.teacher_cfg.initialize_matched_motion,
        )
        with torch.inference_mode():
            self.standing_achieved[ids] = self.current_morphology[ids] == Morphology.BIPED
            self._publish_task_state()
            self.canonical_motion_start[ids] = torch.where(
                self.transition_mask[ids], -1, self._env.common_step_counter,
            )

    def _clear(self, ids):
        super()._clear(ids)
        if hasattr(self, 'canonical_motion_start'):
            self.canonical_motion_start[ids] = -1
        if hasattr(self, 'task_fsm'):
            self.task_fsm.request(ids, self.standing_mode[ids])
            self._publish_task_state()

    def reset(self, env_ids):
        result = super().reset(env_ids)
        ids = self._ids(env_ids)
        low, high = self.teacher_cfg.biped_gait_frequency_range
        self._biped_gait_frequency[ids] = torch.empty(len(ids), device=self.device).uniform_(low, high)
        self.gait_frequency[ids] = torch.where(
            self.standing_mode[ids], self._biped_gait_frequency[ids], self.teacher_cfg.snake_gait_frequency,
        )
        initial = getattr(self._env, '_omni_initial_standing', None)
        self.initialize_task_state(ids, initial[ids] if initial is not None else None)
        return result

    def _update_command(self, env_ids):
        self._publish_task_state()
        super()._update_command(env_ids)
        if env_ids is None:
            self.task_fsm.update_completion(self.hold_time >= self.cfg.hold_s)
        self._publish_task_state()
        ids = self._ids(env_ids)
        self.gait_frequency[ids] = torch.where(
            self.standing_mode[ids], self._biped_gait_frequency[ids], self.teacher_cfg.snake_gait_frequency,
        )

    def _update_straightening(self, ids):
        self.straightening_mask[ids] = False
        self._straightening_pending[ids] = False

    def training_roles(self, ids):
        if not self.cfg.training or self.pretraining or self.cfg.mode != 'source':
            return torch.full_like(ids, -1)
        return self._training_roles[ids]

    def _sample_velocity(self, ids):
        ids = ids[~self.manual_control[ids]]
        self.requested_velocity[ids] = sample_motion_commands(
            self.standing_mode[ids], self.cfg.velocity_limits,
            axis_min_fraction=self.teacher_cfg.motion_axis_min_fraction,
            stop_probability=self.teacher_cfg.motion_stop_probability,
            x_only_probability=getattr(self.teacher_cfg, 'motion_x_only_probability', None),
        )

    def _resample_command(self, env_ids):
        if self.cfg.training:
            if not self.pretraining and self.cfg.mode == 'source' and self.num_envs >= 10:
                ids = env_ids[~self.manual_control[env_ids]]
                ids = ids[~self.transition_mask[ids] | (self.start_step[ids] == self._env.common_step_counter)]
                roles = self.training_roles(ids)
                previous = self._command[ids, 0].clone()
                self._command[ids, 0] = torch.where((roles == 0) | (roles == 2), 1., -1.)
                self._sample_velocity(ids)
                self.requested_velocity[ids[roles >= 2]] = 0.
                self._clear(ids[self._command[ids, 0] != previous])
                return
            return super()._resample_command(env_ids)
        ids = env_ids[~self.manual_control[env_ids]]
        if not len(ids):
            return
        if self.teacher_cfg.play_random_velocity:
            # Only initialize morphology on reset; timer expiry changes velocity alone.
            initial = ids[self.command_counter[ids] == 0]
            previous = self._command[initial, 0].clone()
            if self.cfg.mode == 'source' and self.teacher_cfg.randomize_play_modes:
                assigned = getattr(self._env, '_omni_play_standing', None)
                standing = assigned[initial] if assigned is not None else torch.ones_like(initial, dtype=torch.bool)
                self._command[initial, 0] = torch.where(standing, 1., -1.)
            else:
                self._command[initial, 0] = -1. if self.cfg.mode == 'snake' else 1.
            self._clear(initial[self._command[initial, 0] != previous])
            self._sample_velocity(ids)
            return
        velocity = getattr(self.cfg, 'play_velocity', (.05, 0., 0.))
        if len(velocity) != 3 or not all(math.isfinite(value) for value in velocity):
            raise ValueError('play_velocity must contain three finite values')
        if self.cfg.mode == 'source' and getattr(self.cfg, 'randomize_play_modes', False):
            assigned = getattr(self._env, '_omni_play_standing', None)
            standing = assigned[ids] if assigned is not None else torch.ones(len(ids), device=self.device, dtype=torch.bool)
            previous = self._command[ids, 0].clone()
            self._command[ids, 0] = torch.where(standing, 1., -1.)
            self.velocity_command[ids] = 0
            self.time_left[ids] = math.inf
            self._clear(ids[self._command[ids, 0] != previous])
        else:
            super()._resample_command(ids)
        limits = constant(self.requested_velocity, self.cfg.velocity_limits)
        self.requested_velocity[ids] = constant(self.requested_velocity, velocity).clamp(-limits, limits)


def privileged_state(env):
    """56维特权状态：构型与任务各一维，兼容旧59维检查点。"""
    term = cast(PrivilegedOmniCommand, mdp.command_term(env))
    encoding = term.teacher_cfg.task_observation_encoding
    if encoding not in ('scalar', 'one_hot'):
        raise ValueError('task_observation_encoding must be scalar or one_hot')
    state = mdp.critic_state(env)
    if encoding == 'scalar':
        state = torch.cat((
            state[..., :38],
            term.current_morphology.to(state.dtype)[..., None],
            term.current_task.to(state.dtype)[..., None],
            state[..., 43:],
        ), -1)
    values = (state, mdp.tracking_velocity(env), term.requested_velocity)
    return torch.nan_to_num(torch.cat(values, -1), nan=0., posinf=0., neginf=0.)


def training_skill_complete(env):
    """Recycle successful dedicated transformation trials without cutting motion trials."""
    term = mdp.command_term(env)
    if not isinstance(term, PrivilegedOmniCommand):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if not term.cfg.training or term.pretraining or term.cfg.mode != 'source':
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return (term._training_roles >= 2) & ~term.transition_mask
