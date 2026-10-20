"""Archived transformation failure lifecycle isolated from locomotion and legacy state."""
from dataclasses import dataclass

import torch

from ..transformation import commanded, robust
from ..transformation import mdp as transform
from ..transformation import snake_to_biped as standing
from . import mdp


@dataclass(frozen=True)
class TransitionFailureCfg:
    fall_retry_arm_height: float = .05
    fall_retry_height: float = .1
    fall_retry_drop_threshold: float = .01
    fall_retry_hold_s: float = .1
    early_contact_arm_height: float = .05
    early_contact_motion_threshold: float = .01
    early_contact_drop_threshold: float = .01
    early_contact_min_down_speed: float = .03
    stagnation_height_tolerance: float = .03
    stagnation_joint_tolerance: float = .25
    stagnation_joint_speed: float = .03
    stagnation_base_speed: float = .01
    stagnation_angular_speed: float = .05
    stagnation_hold_s: float = 1.
    stagnation_grace_s: float = .3


ARCHIVE_FAILURE_CFG = TransitionFailureCfg()


@dataclass
class TransitionFailureMemory:
    peak: torch.Tensor
    low_time: torch.Tensor
    still_time: torch.Tensor
    motion: torch.Tensor
    had_contact: torch.Tensor
    separated: torch.Tensor
    early_contact: torch.Tensor
    start_step: torch.Tensor
    standing_mode: torch.Tensor
    last_step: torch.Tensor


def transition_memory(env):
    if not hasattr(env, '_teacher_transition_failure_memory'):
        term = mdp.command_term(env)
        zeros = torch.zeros(env.num_envs, device=env.device)
        flags = zeros.bool()
        env._teacher_transition_failure_memory = TransitionFailureMemory(
            torch.full_like(zeros, -torch.inf), zeros.clone(), zeros.clone(),
            flags.clone(), flags.clone(), flags.clone(), flags.clone(),
            torch.full_like(term.start_step, -1), term.standing_mode.clone(),
            torch.full_like(term.start_step, -1),
        )
    return env._teacher_transition_failure_memory


def reset_transition_failure_memory(env, env_ids):
    """Reset private detector history only; never modify the robot or command."""
    memory = transition_memory(env)
    ids = slice(None) if env_ids is None else env_ids
    with torch.inference_mode():
        memory.peak[ids] = -torch.inf
        memory.low_time[ids] = 0
        memory.still_time[ids] = 0
        for name in ('motion', 'had_contact', 'separated', 'early_contact'):
            getattr(memory, name)[ids] = False
        memory.start_step[ids] = -1
        memory.last_step[ids] = -1


def transition_failure(env):
    """Reuse fall, meaningful recontact, and abandoned-exploration conditions.

    Adaptation: only genuine transitions are eligible. Canonical locomotion and
    manual play cannot trigger these archived transformation-only detectors.
    The independent cache restarts on every episode or target change.
    """
    cfg = ARCHIVE_FAILURE_CFG
    term = mdp.command_term(env)
    memory = transition_memory(env)
    active = term.transition_mask & ~term.manual_control
    if not term.cfg.training:
        active = torch.zeros_like(active)
    changed = (memory.start_step != term.start_step) | (memory.standing_mode != term.standing_mode)
    fresh = (memory.last_step != env.common_step_counter) | changed
    restart = changed | ~active
    data, base_id = transform._base(env)
    height = standing.base_height(env)
    peak = torch.maximum(torch.where(restart, height, memory.peak), height)
    motion = torch.where(restart, False, memory.motion) | (transform.commanded_target(env).abs().amax(-1) >= cfg.early_contact_motion_threshold)
    contact = robust.base_ground_contact(env)
    had_contact = torch.where(restart, False, memory.had_contact)
    separated = torch.where(restart, False, memory.separated) | (motion & had_contact & ~contact)
    drop = peak - height
    low = active & term.standing_mode & (peak >= cfg.fall_retry_arm_height) & (drop >= cfg.fall_retry_drop_threshold) & (height < cfg.fall_retry_height)
    low_time = torch.where(low, torch.where(restart, 0., memory.low_time) + env.step_dt, 0.)
    meaningful = (peak >= cfg.early_contact_arm_height) & (drop >= cfg.early_contact_drop_threshold)
    impact = separated | (standing.base_vertical_velocity(env) <= -cfg.early_contact_min_down_speed)
    detected = active & term.standing_mode & motion & contact & meaningful & impact
    early = (torch.where(restart, False, memory.early_contact) | detected) & active
    goal_height = torch.where(term.standing_mode, env.cfg.standing_base_height, standing.SNAKE_ROOT_POS[2])
    near_goal = (height - goal_height).abs() < cfg.stagnation_height_tolerance
    near_goal &= (data.joint_pos - commanded.goal_joints(env)).abs().amax(-1) < cfg.stagnation_joint_tolerance
    quiet = data.joint_vel.square().mean(-1).sqrt() < cfg.stagnation_joint_speed
    quiet &= data.body_link_lin_vel_w[:, base_id].norm(dim=-1) < cfg.stagnation_base_speed
    quiet &= data.body_link_ang_vel_w[:, base_id].norm(dim=-1) < cfg.stagnation_angular_speed
    eligible = (env.common_step_counter - term.start_step) * env.step_dt >= cfg.stagnation_grace_s
    still = active & eligible & quiet & ~near_goal
    still_time = torch.where(still, torch.where(restart, 0., memory.still_time) + env.step_dt, 0.)
    with torch.inference_mode():
        for current, updated in ((memory.peak, peak), (memory.low_time, low_time), (memory.still_time, still_time),
                                 (memory.motion, motion), (memory.had_contact, had_contact | contact),
                                 (memory.separated, separated), (memory.early_contact, early)):
            current.copy_(torch.where(fresh, updated, current))
        memory.start_step.copy_(term.start_step)
        memory.standing_mode.copy_(term.standing_mode)
        memory.last_step.fill_(env.common_step_counter)
    fallen = term.standing_mode & (memory.low_time >= cfg.fall_retry_hold_s - 1e-6)
    abandoned = memory.still_time > cfg.stagnation_hold_s + 1e-6
    return active & (fallen | memory.early_contact | abandoned)
