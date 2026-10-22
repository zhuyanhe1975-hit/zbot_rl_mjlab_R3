"""Opt-in student PPO completion, continuation trials and command curriculum."""

import itertools
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import torch

from . import mdp
from .guided_transitions import GuidedOmniCommand, GuidedOmniCommandCfg
from .student_observations import GuidedStudentEnvCfg


@dataclass
class StudentPPOEnvCfg(GuidedStudentEnvCfg):
    student_ppo_completion_gate: bool = True
    completion_enter_tolerance: float = .15
    completion_exit_tolerance: float = .20
    student_post_transition_seconds: float = 3.
    student_continuation_fraction: float = .5


@dataclass(kw_only=True)
class StudentPPOCommandCfg(GuidedOmniCommandCfg):
    balanced_motion_commands: bool = False
    snake_single_axis_iterations: int = 20
    snake_two_axis_iterations: int = 20
    snake_command_initial_scale: float = .5
    snake_command_ramp_iterations: int = 100

    def build(self, env):
        return StudentPPOCommand(self, env)


def _validate(env):
    cfg = env.cfg
    enter, leave = cfg.completion_enter_tolerance, cfg.completion_exit_tolerance
    if not (math.isfinite(enter) and math.isfinite(leave) and 0 < enter <= leave):
        raise ValueError('completion tolerances must be finite and 0 < entry <= exit')
    if not math.isfinite(cfg.student_post_transition_seconds) or cfg.student_post_transition_seconds < 0:
        raise ValueError('student post-transition duration must be finite and nonnegative')
    if not math.isfinite(cfg.student_continuation_fraction) or not 0 <= cfg.student_continuation_fraction <= 1:
        raise ValueError('student continuation fraction must be in [0, 1]')


def _state(env):
    term = mdp.command_term(env)
    state = getattr(env, '_student_ppo_state', None)
    if state is None:
        _validate(env)
        with torch.inference_mode(False):
            zero = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            state = SimpleNamespace(
                candidate=zero.clone(), handoff=zero.clone(), continuation=zero.clone(),
                completed_step=torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
                request_step=term.start_step.clone(), target=term.standing_mode.clone(),
                episode=getattr(env, 'episode_length_buf', term.start_step).clone(),
                clock=env.common_step_counter,
            )
        env._student_ppo_state = state
    episode = getattr(env, 'episode_length_buf', term.start_step)
    stale = (state.request_step != term.start_step) | (state.target != term.standing_mode)
    stale |= episode < state.episode
    if env.common_step_counter < state.clock:
        stale = torch.ones_like(stale)
    state.candidate.masked_fill_(stale, False)
    state.handoff.masked_fill_(stale, False)
    state.continuation.masked_fill_(stale, False)
    state.completed_step.masked_fill_(stale, -1)
    term.hold_time.masked_fill_(stale, 0.)
    state.request_step.copy_(term.start_step)
    state.target.copy_(term.standing_mode)
    state.episode.copy_(episode)
    state.clock = env.common_step_counter
    return state


def reset_student_ppo_state(env, env_ids=None):
    """Reset only selected episode memory; never write physical robot state."""
    state = _state(env)
    ids = slice(None) if env_ids is None else env_ids
    state.candidate[ids] = False
    state.handoff[ids] = False
    state.continuation[ids] = False
    state.completed_step[ids] = -1
    mdp.command_term(env).hold_time[ids] = 0.


def student_descent_ready(env, geometric_ready):
    """Require straight actual AND target entry, with a wider exit threshold.

    The command FSM owns the existing continuous hold timer. Geometry/contact
    remains mandatory on every query; no elapsed time is accumulated here.
    """
    state = _state(env)
    term = mdp.command_term(env)
    actual = env.scene['robot'].data.joint_pos.abs().amax(-1)
    target = env.action_manager.get_term('joint_pos').commanded_target.abs().amax(-1)
    maximum = torch.maximum(actual, target)
    finite = torch.isfinite(maximum)
    descending = ~term.standing_mode & term.transition_mask
    entered = maximum <= env.cfg.completion_enter_tolerance
    retained = state.candidate & (maximum <= env.cfg.completion_exit_tolerance)
    state.candidate.copy_(descending & geometric_ready & finite & (entered | retained))
    return geometric_ready & (~descending | state.candidate)


class StudentPPOCommand(GuidedOmniCommand):
    def __init__(self, cfg: StudentPPOCommandCfg, env):
        for name in ('snake_single_axis_iterations', 'snake_two_axis_iterations', 'snake_command_ramp_iterations'):
            value = getattr(cfg, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not math.isfinite(cfg.snake_command_initial_scale) or not 0 < cfg.snake_command_initial_scale <= 1:
            raise ValueError('snake_command_initial_scale must be in (0, 1]')
        self._category_stage = -1
        self._category_cursor = 0
        self._balanced_categories = {}
        self._balanced_cursors = {}
        super().__init__(cfg, env)

    def _clear(self, ids):
        super()._clear(ids)
        if getattr(self._env, '_student_ppo_state', None) is not None:
            reset_student_ppo_state(self._env, ids)

    def _sample_velocity(self, ids):
        cfg = cast(StudentPPOCommandCfg, self.cfg)
        ids = ids[~self.manual_control[ids]]
        biped = ids[self.standing_mode[ids]]
        snake = ids[~self.standing_mode[ids]]
        # Validation overwrites sampled commands; preserve its original RNG
        # sequence so enabling training balance cannot alter validation gait.
        if cfg.balanced_motion_commands and cfg.training:
            self._sample_balanced_velocity(biped, standing=True)
            self._sample_balanced_velocity(snake, standing=False)
            return
        # Preserve the original biped sampling implementation and RNG order.
        super()._sample_velocity(biped)
        if not len(snake):
            return
        iteration = max(0, int(getattr(self._env, '_student_rl_iteration', 0)))
        stage = 3 if not cfg.training else 1 if iteration < cfg.snake_single_axis_iterations else (
            2 if iteration < cfg.snake_single_axis_iterations + cfg.snake_two_axis_iterations else 3
        )
        if stage != self._category_stage:
            categories = [item for item in itertools.product((-1., 0., 1.), repeat=3)
                          if sum(value != 0 for value in item) <= stage]
            with torch.inference_mode(False):
                values = self.requested_velocity.new_tensor(categories)
                self._categories = values[torch.randperm(len(values), device=self.device)]
            self._category_stage, self._category_cursor = stage, 0
        count = len(self._categories)
        index = (torch.arange(len(snake), device=self.device) + self._category_cursor) % count
        self._category_cursor = (self._category_cursor + len(snake)) % count
        progress = 1. if cfg.snake_command_ramp_iterations == 0 else min(1., iteration / cfg.snake_command_ramp_iterations)
        scale = 1. if not cfg.training else cfg.snake_command_initial_scale + (1 - cfg.snake_command_initial_scale) * progress
        magnitude = cfg.snake_motion_axis_min_fraction + torch.rand(len(snake), 3, device=self.device) * (1 - cfg.snake_motion_axis_min_fraction)
        self.requested_velocity[snake] = self._categories[index] * magnitude * self.requested_velocity.new_tensor(cfg.snake_velocity_limits) * scale
        self.gait_frequency[snake] = cfg.snake_gait_frequency

    def _sample_balanced_velocity(self, ids, *, standing):
        """Cycle validation directions independently for each morphology."""
        if not len(ids):
            return
        cfg = cast(StudentPPOCommandCfg, self.cfg)
        if standing not in self._balanced_categories:
            with torch.inference_mode(False):
                categories = self.requested_velocity.new_tensor((
                    (0., 0., 0.), (1., 0., 0.), (-1., 0., 0.),
                    (0., 1., 0.), (0., -1., 0.), (0., 0., 1.), (0., 0., -1.),
                    (1., 1., 1.), (-1., -1., -1.),
                ))
                self._balanced_categories[standing] = categories[torch.randperm(9, device=self.device)]
            self._balanced_cursors[standing] = 0
        index = (torch.arange(len(ids), device=self.device) + self._balanced_cursors[standing]) % 9
        self._balanced_cursors[standing] = (self._balanced_cursors[standing] + len(ids)) % 9
        magnitude = self.requested_velocity.new_tensor((.08, .05, .3)) * (
            1 + .5 * torch.rand(len(ids), 3, device=self.device)
        )
        command = self._balanced_categories[standing][index] * magnitude
        if standing:
            ranges = command.new_tensor(cfg.biped_velocity_ranges)
            command = command.clamp(min=ranges[:, 0], max=ranges[:, 1])
            self.gait_frequency[ids] = self._biped_gait_frequency[ids]
        else:
            limits = command.new_tensor(cfg.snake_velocity_limits)
            iteration = max(0, int(getattr(self._env, '_student_rl_iteration', 0)))
            progress = 1. if cfg.snake_command_ramp_iterations == 0 else min(1., iteration / cfg.snake_command_ramp_iterations)
            scale = 1. if not cfg.training else cfg.snake_command_initial_scale + (1 - cfg.snake_command_initial_scale) * progress
            command = command.clamp(min=-limits, max=limits) * scale
            self.gait_frequency[ids] = cfg.snake_gait_frequency
        self.requested_velocity[ids] = command

    def _resample_command(self, env_ids):
        state = getattr(self._env, '_student_ppo_state', None)
        if state is not None:
            # Successful transform roles must retain their destination command
            # until the continuation termination, including explicit resamples.
            keep = state.handoff[env_ids] & state.continuation[env_ids]
            self.time_left[env_ids[keep]] = math.inf
            env_ids = env_ids[~keep]
        super()._resample_command(env_ids)


def student_training_skill_complete(env):
    """Mix immediate successful resets with actual post-handoff locomotion."""
    term = cast(StudentPPOCommand, mdp.command_term(env))
    empty = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    # Play must neither terminate nor sample/mutate continuation commands.
    if not term.cfg.training or term.pretraining or term.cfg.mode != 'source':
        return empty
    state = _state(env)
    eligible = (term._training_roles >= 2) & ~term.transition_mask & ~term.manual_control
    fresh = eligible & ~state.handoff
    ids = torch.where(fresh)[0]
    if len(ids):
        state.handoff[ids] = True
        state.completed_step[ids] = env.common_step_counter
        state.continuation[ids] = torch.rand(len(ids), device=env.device) < env.cfg.student_continuation_fraction
        moving = ids[state.continuation[ids]]
        term._sample_velocity(moving)
        stopped = moving[term.requested_velocity[moving].norm(dim=-1) <= 1e-6]
        term.requested_velocity[stopped, 0] = .08
        term.time_left[moving] = math.inf
    elapsed = (env.common_step_counter - state.completed_step) * env.step_dt
    complete = ~state.continuation | (elapsed >= env.cfg.student_post_transition_seconds - 1e-6)
    return eligible & state.handoff & complete
