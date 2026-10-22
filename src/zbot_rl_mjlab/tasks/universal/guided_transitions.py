"""Guided-only mirrored transformations and physically executed stand-up retries."""

import math
from dataclasses import dataclass
from typing import cast

import torch

from ..transformation.actions import (
    JointTargetRateAction,
    expand_mirror,
    project_mirror,
)
from . import mdp
from .actions import SnakeCrawlJointTargetRateAction, SnakeCrawlJointTargetRateActionCfg
from .privileged import (
    PrivilegedOmniCommand,
    PrivilegedOmniCommandCfg,
    privileged_state,
)
from .task_fsm import TaskState
from .teacher_sampling import sample_motion_commands
from .teacher_transition import (
    ARCHIVE_FAILURE_CFG,
    reset_transition_failure_memory,
    transition_failure,
)


@dataclass(kw_only=True)
class GuidedOmniCommandCfg(PrivilegedOmniCommandCfg):
    aligned_teacher_sampling: bool = False
    biped_velocity_ranges: tuple[tuple[float, float], ...] = ((-.3, .5), (-.2, .2), (-.5, .5))
    snake_velocity_limits: tuple[float, float, float] = (.2, .2, .5)
    biped_resampling_time_range: tuple[float, float] = (5., 10.)
    snake_resampling_time_range: tuple[float, float] = (5., 9.)
    snake_motion_axis_min_fraction: float = .4
    snake_motion_stop_probability: float = .125
    guided_transition_rules: bool = True
    straightening_joint_tolerance: float = .08
    straightening_hold_s: float = .1
    retry_motion_threshold: float = .25

    def build(self, env):
        return GuidedOmniCommand(self, env)


class GuidedOmniCommand(PrivilegedOmniCommand):
    """Keep ASCENDING latched while returning to a straight pose before retrying.

    A low bent reset/request starts with straightening. Once released, ordinary
    bending cannot reactivate it: only an armed rise followed by a sustained
    height drop can start another attempt. No robot state is reset here.
    """

    def __init__(self, cfg: GuidedOmniCommandCfg, env):
        if cfg.task_observation_encoding != 'scalar':
            raise ValueError('guided transition rules require scalar task observations')
        if len(cfg.biped_velocity_ranges) != 3:
            raise ValueError('biped_velocity_ranges must contain three low/high pairs')
        for name, ranges in (
            ('biped_velocity_ranges', cfg.biped_velocity_ranges),
            ('biped_resampling_time_range', (cfg.biped_resampling_time_range,)),
            ('snake_resampling_time_range', (cfg.snake_resampling_time_range,)),
        ):
            for pair in ranges:
                if len(pair) != 2 or not all(math.isfinite(x) for x in pair) or pair[0] > pair[1]:
                    raise ValueError(f'{name} must contain finite ordered low/high pairs')
                if name != 'biped_velocity_ranges' and pair[0] <= 0:
                    raise ValueError(f'{name} must be positive')
        if len(cfg.snake_velocity_limits) != 3 or not all(math.isfinite(x) and x > 0 for x in cfg.snake_velocity_limits):
            raise ValueError('snake_velocity_limits must contain three finite positive values')
        if not math.isfinite(cfg.snake_motion_axis_min_fraction) or not 0 < cfg.snake_motion_axis_min_fraction <= 1:
            raise ValueError('snake_motion_axis_min_fraction must be in (0, 1]')
        if not math.isfinite(cfg.snake_motion_stop_probability) or not 0 <= cfg.snake_motion_stop_probability <= 1:
            raise ValueError('snake_motion_stop_probability must be in [0, 1]')
        for name in ('straightening_joint_tolerance', 'straightening_hold_s', 'retry_motion_threshold'):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        super().__init__(cfg, env)
        # Keyboard requests can arrive outside viewer inference_mode.
        with torch.inference_mode(False):
            self.retry_attempts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._retry_peak = torch.full((self.num_envs,), -torch.inf, device=self.device)
            self._retry_low_time = torch.zeros(self.num_envs, device=self.device)
            self._retry_armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._straightening_time = torch.zeros(self.num_envs, device=self.device)
            self._rules_last_step = torch.full_like(self.start_step, -1)

    def _sample_velocity(self, ids):
        cfg = cast(GuidedOmniCommandCfg, self.cfg)
        if not cfg.aligned_teacher_sampling:
            return super()._sample_velocity(ids)
        ids = ids[~self.manual_control[ids]]
        biped = ids[self.standing_mode[ids]]
        snake = ids[~self.standing_mode[ids]]
        ranges = self.requested_velocity.new_tensor(cfg.biped_velocity_ranges)
        self.requested_velocity[biped] = ranges[:, 0] + torch.rand(
            len(biped), 3, device=self.device,
        ) * (ranges[:, 1] - ranges[:, 0])
        self.requested_velocity[snake] = sample_motion_commands(
            torch.zeros(len(snake), dtype=torch.bool, device=self.device),
            cfg.snake_velocity_limits,
            axis_min_fraction=cfg.snake_motion_axis_min_fraction,
            stop_probability=cfg.snake_motion_stop_probability,
            x_only_probability=None,
        )
        self.gait_frequency[ids] = torch.where(
            self.standing_mode[ids], self._biped_gait_frequency[ids], cfg.snake_gait_frequency,
        )

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        cfg = cast(GuidedOmniCommandCfg, self.cfg)
        if not cfg.aligned_teacher_sampling:
            return
        self.time_left[env_ids[self.manual_control[env_ids]]] = math.inf
        # CommandTerm sets the shared timer before this callback. Replace only
        # active automatic timers, retaining manual/play infinite timers and FSM.
        ids = env_ids[~self.manual_control[env_ids] & torch.isfinite(self.time_left[env_ids])]
        for standing, interval in (
            (True, cfg.biped_resampling_time_range), (False, cfg.snake_resampling_time_range),
        ):
            selected = ids[self.standing_mode[ids] == standing]
            self.time_left[selected] = torch.empty(len(selected), device=self.device).uniform_(*interval)

    def set_velocity(self, vx, vy, wz, env_ids=None):
        cfg = cast(GuidedOmniCommandCfg, self.cfg)
        if not cfg.aligned_teacher_sampling:
            return super().set_velocity(vx, vy, wz, env_ids)
        values = (vx, vy, wz)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('速度指令必须为有限值')
        ids = self._ids(env_ids)
        ranges = self.requested_velocity.new_tensor(cfg.biped_velocity_ranges)
        limits = self.requested_velocity.new_tensor(cfg.snake_velocity_limits)
        # standing_mode is the requested target during a transition, so pending
        # commands use the destination's bounds while effective velocity stays gated.
        standing = self.standing_mode[ids, None]
        low = torch.where(standing, ranges[:, 0], -limits)
        high = torch.where(standing, ranges[:, 1], limits)
        self.requested_velocity[ids] = self.requested_velocity.new_tensor(values).clamp(min=low, max=high)
        self.manual_control[ids] = True
        self.time_left[ids] = math.inf

    def _update_command(self, env_ids):
        cfg = cast(GuidedOmniCommandCfg, self.cfg)
        previous_tasks = self.current_task.clone() if not cfg.training and env_ids is None else None
        super()._update_command(env_ids)
        if previous_tasks is not None:
            # A completed task may change BASE/site axes. Rebase heading once.
            self.heading_initialized &= self.current_task == previous_tasks

        if not cfg.aligned_teacher_sampling or env_ids is not None:
            return
        # Native UniformVelocityCommand is direct; retain the native snake
        # low-pass filter and the inherited zero target throughout transitions.
        biped = (self.current_task == TaskState.BIPED_MOTION) & self.standing_mode & ~self.transition_mask
        self.velocity_command.copy_(torch.where(biped[:, None], self.requested_velocity, self.velocity_command))

    def _clear(self, ids):
        super()._clear(ids)
        if hasattr(self, 'retry_attempts'):
            self.retry_attempts[ids] = 0
            self._reset_attempt(ids)
            self._straightening_time[ids] = 0
            self._rules_last_step[ids] = -1

    def set_target(self, target, env_ids=None):
        ids = self._ids(env_ids)
        changed = ids[self._command[ids, 0] != (1. if target == 'stand' else -1.)]
        # Only an explicit stand request can recover a fallen walking robot.
        recover = torch.empty(0, dtype=torch.long, device=self.device)
        if target == 'stand':
            ready = mdp.configuration_ready(self._env, relaxed=True)
            recover = ids[(self.current_task[ids] == TaskState.BIPED_MOTION) & ~ready[ids]]
        super().set_target(target, env_ids)
        self._clear(recover)
        self.task_fsm.reset(
            recover, torch.ones_like(recover, dtype=torch.bool),
            torch.ones_like(recover, dtype=torch.bool), force_transition=True,
        )
        # _clear published the old BIPED state before the explicit transition.
        # A fallen robot must regain standing confirmation in this new attempt.
        self.standing_achieved[recover] = False
        self._publish_task_state()
        if not self.cfg.training:
            selected = torch.unique(torch.cat((changed, recover)))
            cfg = cast(GuidedOmniCommandCfg, self.cfg)
            if cfg.aligned_teacher_sampling:
                ranges = self.requested_velocity.new_tensor(cfg.biped_velocity_ranges)
                limits = self.requested_velocity.new_tensor(cfg.snake_velocity_limits)
                standing = self.standing_mode[selected, None]
                low = torch.where(standing, ranges[:, 0], -limits)
                high = torch.where(standing, ranges[:, 1], limits)
                self.requested_velocity[selected] = self.requested_velocity[selected].clamp(min=low, max=high)
            self.velocity_command[selected] = 0.
            self.gait_frequency[selected] = torch.where(
                self.standing_mode[selected], self._biped_gait_frequency[selected], cfg.snake_gait_frequency,
            )
            self.heading_initialized[selected] = False

    def _reset_attempt(self, ids):
        self._retry_peak[ids] = -torch.inf
        self._retry_low_time[ids] = 0
        self._retry_armed[ids] = False

    def _update_straightening(self, ids):
        # DeploymentCommand invokes this exactly once before the normal FSM
        # completion checks. A per-row stamp also protects duplicate callers.
        if not hasattr(self, 'retry_attempts'):
            return
        cfg = cast(GuidedOmniCommandCfg, self.cfg)
        failure = ARCHIVE_FAILURE_CFG
        chosen = torch.zeros_like(self.straightening_mask)
        chosen[ids] = True
        fresh = chosen & (self._rules_last_step != self._env.common_step_counter)
        active = self.current_task == TaskState.ASCENDING
        data = self._env.scene['robot'].data
        joints = data.joint_pos
        target = self._env.action_manager.get_term('joint_pos').commanded_target
        height = mdp.base_height(self._env)
        maximum = torch.maximum(joints.abs().amax(-1), target.abs().amax(-1))
        pending = fresh & self._straightening_pending
        begin = pending & active & (height < failure.fall_retry_height) & (maximum > cfg.straightening_joint_tolerance)
        self.straightening_mask |= begin
        self._straightening_pending &= ~fresh
        self.straightening_mask &= active
        attempting = fresh & active & ~self.straightening_mask
        peak = torch.maximum(self._retry_peak, height)
        self._retry_peak.copy_(torch.where(attempting, peak, self._retry_peak))
        self._retry_armed |= attempting & (maximum >= cfg.retry_motion_threshold) & (height >= failure.fall_retry_arm_height)
        dropped = (self._retry_peak - height) >= failure.fall_retry_drop_threshold
        failed_low = attempting & self._retry_armed & dropped & (height < failure.fall_retry_height)
        low_time = torch.where(failed_low, self._retry_low_time + self._env.step_dt, 0.)
        self._retry_low_time.copy_(torch.where(fresh, low_time, self._retry_low_time))
        retry = failed_low & (self._retry_low_time >= failure.fall_retry_hold_s - 1e-6)
        self.straightening_mask |= retry
        self.retry_attempts += retry.long()
        self._straightening_time.masked_fill_(begin | retry, 0.)
        straight = (maximum <= cfg.straightening_joint_tolerance) & (data.joint_vel.abs().amax(-1) <= failure.stagnation_joint_speed)
        held = torch.where(straight & self.straightening_mask, self._straightening_time + self._env.step_dt, 0.)
        self._straightening_time.copy_(torch.where(fresh, held, self._straightening_time))
        release = fresh & self.straightening_mask & (self._straightening_time >= cfg.straightening_hold_s - 1e-6)
        self.straightening_mask &= ~release
        # Pause the transformation timeout throughout physical recovery and
        # start a fresh attempt only after both target AND measured pose settle.
        restarting = fresh & (self.straightening_mask | release)
        self.start_step.masked_fill_(restarting, self._env.common_step_counter)
        self.hold_time.masked_fill_(restarting, 0.)
        self._reset_attempt(torch.where(restarting)[0])
        reset_transition_failure_memory(self._env, torch.where(restarting)[0])
        self._rules_last_step.masked_fill_(fresh, self._env.common_step_counter)


@dataclass(kw_only=True)
class GuidedTransitionActionCfg(SnakeCrawlJointTargetRateActionCfg):
    def build(self, env):
        return GuidedTransitionAction(self, env)


class GuidedTransitionAction(SnakeCrawlJointTargetRateAction):
    """Mirror transition targets; gracefully remove asymmetry at the speed limit."""

    def process_actions(self, actions):
        previous = self.commanded_target.clone()
        super().process_actions(actions)
        command = cast(GuidedOmniCommand, self._env.command_manager.get_term('transform'))
        limits = self._target_limits()
        left, right = limits[:, :3], limits[:, 3:].flip(1)
        paired_low = torch.maximum(left[..., 0], -right[..., 1])
        paired_high = torch.minimum(left[..., 1], -right[..., 0])
        coordinates = project_mirror(self.commanded_target).clamp(min=paired_low, max=paired_high)
        desired = expand_mirror(coordinates)
        desired = torch.where(command.straightening_mask[:, None], 0., desired)
        delta = self.rate_cfg.max_target_speed * self._env.step_dt
        constrained = previous + (desired - previous).clamp(-delta, delta)
        # A newly entered phase may narrow target bounds around a current
        # target outside that interval. Approach those desired bounds at the
        # speed limit while always respecting the physical/static bounds.
        physical = JointTargetRateAction._target_limits(self)
        constrained = constrained.clamp(min=physical[..., 0], max=physical[..., 1])
        self._processed_actions.copy_(torch.where(command.transition_mask[:, None], constrained, self.commanded_target))


def guided_privileged_state(env):
    """Use a fractional ASCENDING code to expose the recovery substage in 56D."""
    state = privileged_state(env)
    command = cast(GuidedOmniCommand, mdp.command_term(env))
    recovering = command.straightening_mask & (command.current_task == TaskState.ASCENDING)
    state[:, 39] = torch.where(recovering, 3.25, state[:, 39])
    return state


def guided_transition_failure(env):
    """Ascent failures retry in the command FSM; preserve descent termination.

    Unsafe/numeric/workspace and episode timeout terms are separate and remain
    enabled by the experiment configuration.
    """
    command = mdp.command_term(env)
    return transition_failure(env) & (command.phase == TaskState.DESCENDING)
