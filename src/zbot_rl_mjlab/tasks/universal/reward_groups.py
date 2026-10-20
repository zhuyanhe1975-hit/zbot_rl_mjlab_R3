"""Reward-component streams for the isolated multi-critic teacher experiment.

The physical reward is unchanged. Capture its already-computed components
before auto-reset so PPO does not recompute stateful rewards on reset states.
"""

from typing import Any

import torch

from .observation_layout import teacher_task_ids

GROUP_NAMES = ('motion_tracking', 'configuration', 'contact_gait', 'regularization')
REWARD_GROUPS = {
    'motion_tracking': ('velocity_tracking', 'standstill'),
    'configuration': (
        'mode_posture', 'transition_progress', 'joint_standing',
        'base_height_progress', 'base_ascent_velocity', 'sole_ground_alignment',
        'reset_failure', 'stand_finish_joint_speed', 'descent_ground_impact',
    ),
    'contact_gait': (
        'soft_landing', 'swing_clearance', 'support_foot_slip',
        'foot_nonparallel_ground',
    ),
    'regularization': (
        'action_rate', 'biped_upright', 'joint_limits', 'joint_acceleration',
        'joint_energy', 'snake_vertical_speed', 'snake_joint_deviation',
        'snake_target_saturation',
    ),
}


def group_mask_from_tasks(tasks: torch.Tensor) -> torch.Tensor:
    """Semantic activity, including active groups whose instantaneous reward is zero."""
    return torch.stack((
        (tasks == 0) | (tasks == 2),
        (tasks == 1) | (tasks == 3),
        tasks == 0,
        torch.ones_like(tasks, dtype=torch.bool),
    ), dim=-1)


def reward_group_mask(observations: torch.Tensor) -> torch.Tensor:
    return group_mask_from_tasks(teacher_task_ids(observations))


def reward_assignment(names: list[str], *, device: str | torch.device) -> torch.Tensor:
    """Require exactly one destination for every active reward component."""
    owners: dict[str, int] = {}
    for index, group in enumerate(GROUP_NAMES):
        for name in REWARD_GROUPS[group]:
            if name in owners:
                raise ValueError(f'reward {name!r} appears in multiple groups')
            owners[name] = index
    if len(names) != len(set(names)):
        raise ValueError('active reward names must be unique')
    unknown = set(names) - owners.keys()
    if unknown:
        raise ValueError(f'ungrouped teacher rewards: {sorted(unknown)}')
    assignment = torch.zeros((len(names), len(GROUP_NAMES)), device=device)
    for row, name in enumerate(names):
        assignment[row, owners[name]] = 1.
    return assignment


def install_reward_group_capture(env: Any) -> None:
    """Attach only to this environment's manager; leave ordinary PPO environments alone."""
    manager = env.reward_manager
    if getattr(manager, '_teacher_reward_group_capture', False):
        return
    assignment = reward_assignment(manager.active_terms, device=env.device)
    original_compute = manager.compute

    def compute_with_groups(dt: float) -> torch.Tensor:
        total = original_compute(dt)
        # _step_reward contains weighted rates, including the manager's NaN guard.
        # It survives reset, but task masks must be captured before reset/commands.
        scale = dt if manager._scale_by_dt else 1.
        env.extras['teacher_reward_groups'] = (manager._step_reward @ assignment) * scale
        term = env.command_manager.get_term('transform')
        env.extras['teacher_reward_group_mask'] = group_mask_from_tasks(term.current_task)
        return total

    manager.compute = compute_with_groups
    manager._teacher_reward_group_capture = True
