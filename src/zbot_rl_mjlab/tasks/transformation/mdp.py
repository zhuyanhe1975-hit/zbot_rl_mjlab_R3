"""两个目标：全身质心平缓下降与关节角归零。

关节对称仅作为软正则约束；六个策略动作仍彼此独立。
"""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse


class TransformationState:
    def __init__(self, num_envs, device):
        self.hold_time = torch.zeros(num_envs, device=device)
        self.last_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)

    def reset(self, env_ids=None, step=-1):
        ids = slice(None) if env_ids is None else env_ids
        self.hold_time[ids] = 0.0
        self.last_step[ids] = step

    def advance_hold(self, valid, step, dt, *, commit=True):
        """每个控制步仅累计一次满足目标的保持时间。"""
        fresh = self.last_step != step
        held = torch.where(valid, self.hold_time + dt, 0.0)
        self.hold_time.copy_(torch.where(fresh, held, self.hold_time))
        if commit:
            self.last_step.fill_(step)
        return fresh


def state(env):
    if not hasattr(env, "_transformation_state"):
        env._transformation_state = TransformationState(env.num_envs, env.device)
    return env._transformation_state


def reset_transformation(env, env_ids):
    state(env).reset(env_ids, env.common_step_counter)


def _base(env):
    robot = env.scene["robot"]
    return robot.data, robot.find_bodies("base")[0][0]


def body_masses(env):
    return env.sim.model.body_mass[..., env.scene["robot"].indexing.body_ids.long()]


def com_state(env):
    data = env.scene["robot"].data
    masses = body_masses(env)
    height = (data.body_com_pos_w[..., 2] * masses).sum(-1) / masses.sum(-1)
    velocity = (data.body_com_lin_vel_w[..., 2] * masses).sum(-1) / masses.sum(-1)
    return torch.stack((height - env.scene.env_origins[:, 2], velocity), dim=-1)


def projected_gravity(env):
    data, i = _base(env)
    gravity = torch.zeros_like(data.body_link_pos_w[:, i])
    gravity[:, 2] = -1.0
    return quat_apply_inverse(data.body_link_quat_w[:, i], gravity)


def joint_positions(env):
    return env.scene["robot"].data.joint_pos


def commanded_target(env):
    return env.action_manager.get_term("joint_pos").commanded_target


def last_action(env):
    return env.action_manager.get_term("joint_pos").raw_action


def hold_observation(env):
    return (state(env).hold_time / env.cfg.success_hold_s).unsqueeze(-1)


def final_condition(env, joint_tolerance=None):
    tolerance = env.cfg.joint_tolerance if joint_tolerance is None else joint_tolerance
    data, i = _base(env)
    low = (data.body_com_pos_w[..., 2] - env.scene.env_origins[:, 2, None]).amax(-1)
    return (
        (data.joint_pos.abs().amax(-1) < tolerance)
        & (low < env.cfg.final_max_body_height)
        & (data.joint_vel.square().mean(-1).sqrt() < env.cfg.joint_speed_tolerance)
        & (data.body_link_lin_vel_w[:, i].norm(dim=-1) < env.cfg.base_speed_tolerance)
    )


def advance(env):
    task = state(env)
    task.advance_hold(final_condition(env), env.common_step_counter, env.step_dt)


def update_state(env):
    advance(env)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def success(env):
    return (state(env).hold_time >= env.cfg.success_hold_s - 1e-5) & final_condition(
        env
    )


def unsafe(env):
    data, i = _base(env)
    finite = torch.isfinite(data.joint_pos).all(-1) & torch.isfinite(
        data.joint_vel
    ).all(-1)
    finite &= torch.isfinite(data.body_com_pos_w).all(dim=(-1, -2))
    finite &= torch.isfinite(data.body_com_lin_vel_w).all(dim=(-1, -2))
    finite &= torch.isfinite(data.body_link_pos_w[:, i]).all(-1)
    finite &= torch.isfinite(data.body_link_lin_vel_w[:, i]).all(-1)
    finite &= torch.isfinite(data.body_link_quat_w[:, i]).all(-1)
    offset = data.body_link_pos_w[:, i, :2] - env.scene.env_origins[:, :2]
    return (
        ~finite
        | (com_state(env)[:, 1] < -env.cfg.max_downward_com_speed)
        | (offset.norm(dim=-1) > env.cfg.workspace_radius)
    )


def joint_zero_reward(env):
    return torch.nan_to_num(torch.exp(-joint_positions(env).square().mean(-1)), nan=0.0)


def com_descent_velocity_reward(env, target_speed=0.03):
    """静止时为零，接近平缓下降时为正，上升或快速跌落时为负。"""
    if target_speed <= 0:
        raise ValueError("target_speed must be positive")
    score = 1.0 - ((com_state(env)[:, 1] + target_speed) / target_speed).square()
    return torch.nan_to_num(score, nan=-4.0, posinf=-4.0, neginf=-4.0).clamp(-4.0, 1.0)


def joint_symmetry_cost(env):
    q = joint_positions(env)
    error = (q[:, :3] + q[:, 3:].flip(-1)).square().mean(-1)
    return torch.nan_to_num(error, nan=1000.0, posinf=1000.0).clamp(max=1000.0)


def joint_error_metric(env):
    return joint_positions(env).abs().mean(-1)


def com_height(env):
    return com_state(env)[:, 0]
