"""行走任务的观测、指令、奖励和扰动实现。"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from ..common import PartialBaseImpulse  # noqa: F401  # 供行走扰动调用方使用

_ASSET_FORWARD_SIGN = 1.0


def _base(env):
    robot = env.scene["robot"]
    return robot, robot.find_bodies("base")[0][0]


def sampled_frequency(env, min_frequency=None, max_frequency=None):
    fixed = getattr(env.cfg, "test_frequency", None)
    if fixed is not None:
        current = torch.full((env.num_envs,), fixed, device=env.device)
        env._zbot_step_frequency = current
        return current
    min_frequency = (
        env.cfg.step_frequency_min if min_frequency is None else min_frequency
    )
    max_frequency = (
        env.cfg.step_frequency_max if max_frequency is None else max_frequency
    )
    current = getattr(env, "_zbot_step_frequency", None)
    if current is None or current.shape[0] != env.num_envs:
        current = torch.full((env.num_envs,), min_frequency, device=env.device)
    fresh = env.episode_length_buf == 0
    if fresh.any():
        if min_frequency == max_frequency:
            values = torch.full((int(fresh.sum()),), min_frequency, device=env.device)
        elif getattr(env.cfg, "frequency_sampling", "random") == "sequential":
            values = torch.linspace(
                min_frequency, max_frequency, env.num_envs, device=env.device
            )[fresh]
        else:
            values = torch.empty((int(fresh.sum()),), device=env.device).uniform_(
                min_frequency, max_frequency
            )
        current = current.clone()
        current[fresh] = values
    env._zbot_step_frequency = current
    return current


def phase(env, min_frequency=None, max_frequency=None):
    frequency = sampled_frequency(env, min_frequency, max_frequency)
    return (
        env.episode_length_buf.float() * env.step_dt * frequency * 2.0 * math.pi
    ).remainder(2.0 * math.pi)


def phase_observation(env, min_frequency=None, max_frequency=None):
    p = phase(env, min_frequency, max_frequency)
    return torch.stack((torch.sin(p), torch.cos(p)), dim=-1)


def frequency_observation(env, min_frequency=None, max_frequency=None):
    """将采样得到的回合频率归一化到 [0, 1] 后提供给观测。"""
    min_frequency = (
        env.cfg.step_frequency_min if min_frequency is None else min_frequency
    )
    max_frequency = (
        env.cfg.step_frequency_max if max_frequency is None else max_frequency
    )
    frequency = sampled_frequency(env, min_frequency, max_frequency)
    span = max(max_frequency - min_frequency, 1e-6)
    return ((frequency - min_frequency) / span).unsqueeze(-1)


def base_lin_vel(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_b, left_b, gravity_b = locomotion_frame(q)
    velocity_w = robot.data.body_link_lin_vel_w[:, i]
    forward_w = quat_apply(q, forward_b)
    left_w = quat_apply(q, left_b)
    gravity_w = quat_apply(q, gravity_b)
    return torch.stack(
        (
            (velocity_w * forward_w).sum(-1),
            (velocity_w * left_w).sum(-1),
            (velocity_w * gravity_w).sum(-1),
        ),
        dim=-1,
    )


def forward_velocity_reward(env, scale=0.2):
    """边际收益递减的有界前进速度奖励。"""
    speed = base_lin_vel(env)[:, 0].clamp_min(0.0)
    return 1.0 - torch.exp(-speed / scale)


def lateral_velocity_penalty(env):
    """惩罚速度在水平运动左轴上的投影。"""
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    _, left_b, _ = locomotion_frame(q)
    left_w = quat_apply(q, left_b)
    lateral_velocity = (robot.data.body_link_lin_vel_w[:, i] * left_w).sum(-1)
    return lateral_velocity.square()


class ForwardVelocityVisualizer(UniformVelocityCommand):
    """仅用于查看器的基座前向轴和实测速度箭头。"""

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.0

    def _debug_vis_impl(self, visualizer):
        indices = visualizer.get_env_indices(self.num_envs)
        if not indices:
            return
        robot, i = _base(self._env)
        q = robot.data.body_link_quat_w[:, i]
        forward_b, _, _ = locomotion_frame(q)
        forward_w = quat_apply(q, forward_b).cpu().numpy()
        pos = robot.data.body_link_pos_w[:, i].cpu().numpy()
        speed = base_lin_vel(self._env)[:, 0].detach().cpu().numpy()
        for j in indices:
            origin = pos[j] + np.array([0.0, 0.0, 0.12])
            scale = 0.25
            visualizer.add_arrow(
                origin,
                origin + forward_w[j] * scale,
                color=(0.0, 1.0, 1.0, 0.8),
                width=0.012,
            )
            visualizer.add_arrow(
                origin,
                origin + forward_w[j] * (speed[j] * scale),
                color=(0.1, 1.0, 0.2, 0.9),
                width=0.018,
            )


def command_arrow_segments(origin, forward, left, command, length=0.25):
    """Zbot 坐标系中归一化 (vx, vy, wz) 指令对应的箭头线段。

    返回平面速度箭头与偏航箭头各自的 (start, end)。
    偏航箭头沿竖直方向：按右手定则，正角速度（从上方看逆时针）沿世界 +Z 向上。
    """
    vx, vy, wz = command
    up = np.array([0.0, 0.0, 1.0])
    linear = (origin, origin + (forward * vx + left * vy) * length)
    yaw = (origin, origin + up * wz * length)
    return linear, yaw


@dataclass(kw_only=True)
class VelocityCommandVizCfg(UniformVelocityCommandCfg):
    # 各轴（vx、vy、wz）绘制全长箭头时对应的指令幅值。
    full_scale: tuple[float, float, float] = (0.5, 0.2, 0.5)

    def build(self, env):
        return VelocityCommandVisualizer(self, env)


class VelocityCommandVisualizer(UniformVelocityCommand):
    """仅在查看器中显示的基座上方 (vx, vy, wz) 箭头：蓝色为指令，绿色为实测。

    使用 Zbot 前向／左向轴，而非浮动的 foot_0 根坐标系。
    每轴长度按 cfg.full_scale 归一化，使单位不同的三个分量仍可比较。
    """

    def _debug_vis_impl(self, visualizer):
        indices = visualizer.get_env_indices(self.num_envs)
        if not indices:
            return
        robot, i = _base(self._env)
        q = robot.data.body_link_quat_w[:, i]
        forward_b, left_b, _ = locomotion_frame(q)
        forward = quat_apply(q, forward_b).cpu().numpy()
        left = quat_apply(q, left_b).cpu().numpy()
        pos = robot.data.body_link_pos_w[:, i].cpu().numpy()
        # base_ang_vel 的第三轴朝下；取反后逆时针为正。
        measured = torch.cat(
            (base_lin_vel(self._env)[:, :2], -base_ang_vel(self._env)[:, 2:]), dim=-1
        )
        full = np.asarray(self.cfg.full_scale)
        commanded = self.command.cpu().numpy() / full
        measured = measured.cpu().numpy() / full
        for j in indices:
            origin = pos[j] + np.array([0.0, 0.0, 0.12])
            for values, color, width, lift in (
                (commanded[j], (0.2, 0.4, 1.0, 0.85), 0.012, 0.0),
                (measured[j], (0.1, 1.0, 0.2, 0.9), 0.018, 0.02),
            ):
                lifted = origin + np.array([0.0, 0.0, lift])
                for start, end in command_arrow_segments(
                    lifted, forward[j], left[j], values
                ):
                    if np.linalg.norm(end - start) > 1e-3:
                        visualizer.add_arrow(start, end, color=color, width=width)


def base_ang_vel(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_b, left_b, gravity_b = locomotion_frame(q)
    velocity_w = robot.data.body_link_ang_vel_w[:, i]
    forward_w = quat_apply(q, forward_b)
    left_w = quat_apply(q, left_b)
    gravity_w = quat_apply(q, gravity_b)
    return torch.stack(
        (
            (velocity_w * forward_w).sum(-1),
            (velocity_w * left_w).sum(-1),
            (velocity_w * gravity_w).sum(-1),
        ),
        dim=-1,
    )


def projected_gravity(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    g = torch.zeros((env.num_envs, 3), device=env.device)
    g[:, 2] = -1
    return quat_apply_inverse(q, g)


def upright(env, std=0.45):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    z = torch.zeros((env.num_envs, 3), device=env.device)
    z[:, 2] = 1
    z_w = quat_apply(q, z)
    return torch.exp(-z_w[:, 2].square() / std**2)


def locomotion_frame(quat):
    world_up = torch.zeros_like(quat[..., :3])
    world_up[..., 2] = 1.0
    gravity = -world_up
    body_z = torch.zeros_like(quat[..., :3])
    body_z[..., 2] = 1.0
    body_z_w = quat_apply(quat, body_z)
    body_z_w[..., 2] = 0.0
    body_z_w = body_z_w / body_z_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    forward_w = torch.linalg.cross(gravity, body_z_w)
    forward_w = forward_w * _ASSET_FORWARD_SIGN
    forward_w = forward_w / forward_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left_w = torch.linalg.cross(forward_w, gravity)
    left_w = left_w / left_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (
        quat_apply_inverse(quat, forward_w),
        quat_apply_inverse(quat, left_w),
        quat_apply_inverse(quat, gravity),
    )


def _forward_world(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_b, _, _ = locomotion_frame(q)
    return quat_apply(q, forward_b)


def _yaw_error(env):
    current = _forward_world(env)
    initial = getattr(env, "_zbot_initial_forward_w", None)
    if initial is None or initial.shape != current.shape:
        initial = current.clone()
    fresh = env.episode_length_buf == 0
    initial = torch.where(fresh.unsqueeze(-1), current, initial)
    env._zbot_initial_forward_w = initial
    cross_z = current[:, 0] * initial[:, 1] - current[:, 1] * initial[:, 0]
    dot = (current[:, :2] * initial[:, :2]).sum(-1)
    return torch.atan2(cross_z, dot)


def yaw_error_observation(env):
    return _yaw_error(env).unsqueeze(-1)


def yaw_drift_penalty(env):
    """惩罚相对每个回合初始前进方向的偏航漂移。"""
    return _yaw_error(env).square()


def feet_contact(env, sensor_name="feet_ground_contact"):
    found = env.scene[sensor_name].data.found
    assert found is not None
    return (found > 0).float()


def alternating_foot_phase(
    env, sensor_name="feet_ground_contact", min_frequency=None, max_frequency=None
):
    p = phase(env, min_frequency, max_frequency)
    force = env.scene[sensor_name].data.force
    assert force is not None
    support = force.norm(dim=-1)
    left_force, right_force = support[:, 0], support[:, 1]
    total = left_force + right_force
    measured = (left_force - right_force) / total.clamp_min(1e-6)
    target = torch.sin(p)
    matched = 1.0 - (measured - target).square()
    return matched.clamp_min(0.0) * (total > 1e-3).float()


def swing_clearance(
    env,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
    target=0.05,
    min_frequency=None,
    max_frequency=None,
):
    del sensor_name
    p = phase(env, min_frequency, max_frequency)
    target_support = torch.sin(p)
    # alternating_foot_phase 为正表示 foot_0 支撑，
    # 为负表示 foot_1 支撑，因此另一足必须摆动。
    swing = torch.stack((target_support < 0, target_support > 0), dim=-1).float()
    heights = env.scene[height_sensor_name].data.heights
    return (swing * torch.exp(-((heights - target) / 0.04).square())).mean(-1)


def support_foot_slip(
    env,
    sensor_name="feet_ground_contact",
    site_names=("foot_0_sole", "foot_1_sole"),
    angular_radius=0.05,
):
    """惩罚足部接触时的平移和旋转滑移。"""
    if angular_radius <= 0:
        raise ValueError("Angular radius must be positive")
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    site_ids, _ = robot.find_sites(site_names, preserve_order=True)
    if len(site_ids) != contact.shape[1]:
        raise ValueError("Expected one sole site per contact sensor primary")
    linear_speed_sq = robot.data.site_lin_vel_w[:, site_ids, :2].square().sum(-1)
    yaw_speed_sq = robot.data.site_ang_vel_w[:, site_ids, 2].square()
    speed_sq = linear_speed_sq + angular_radius**2 * yaw_speed_sq
    return (contact * speed_sq).sum(-1)


def soft_landing(env, sensor_name="feet_ground_contact"):
    """惩罚着地峰值力，并记录对应冲量。"""
    sensor = env.scene[sensor_name]
    data = sensor.data
    assert data.force is not None
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    force_norm = data.force.norm(dim=-1)
    force_history = getattr(data, "force_history", None)
    if force_history is not None:
        history_norm = force_history.norm(dim=-1)
        peak_force = history_norm.amax(dim=-1)
        impulse = history_norm.sum(dim=-1) * env.physics_dt
    else:
        peak_force = force_norm
        impulse = force_norm * env.step_dt
    landing = first_contact.float()
    peak_landing_force = peak_force * landing
    landing_impulse = impulse * landing
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {})["Metrics/landing_force_peak"] = (
            peak_landing_force.sum(-1).mean()
        )
        env.extras["log"]["Metrics/landing_impulse"] = landing_impulse.sum(-1).mean()
    return peak_landing_force.sum(-1)


def action_rate_l2(env):
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)


def termination_penalty(env):
    """在奖励按 dt 缩放前返回一次性的终止代价。"""
    return envs_mdp.is_terminated(env) / env.step_dt


def joint_acc_l2(env):
    """惩罚关节加速度绝对值之和。"""
    return env.scene["robot"].data.joint_acc.abs().sum(dim=-1)


def joint_pos_limits(env):
    r = env.scene["robot"].data
    return (r.joint_pos_limits[..., 0] - r.joint_pos).clamp_min(0).square().sum(-1) + (
        r.joint_pos - r.joint_pos_limits[..., 1]
    ).clamp_min(0).square().sum(-1)


def fallen(env, height=0.22, tilt=0.75):
    robot, i = _base(env)
    pos = robot.data.body_link_pos_w[:, i] - env.scene.env_origins
    q = robot.data.body_link_quat_w[:, i]
    z = torch.zeros((env.num_envs, 3), device=env.device)
    z[:, 2] = 1
    z_w = quat_apply(q, z)
    return (pos[:, 2] < height) | (z_w[:, 2].abs() > tilt)


def foot_self_collision_penalty(env, sensor_name="feet_self_contact", scale=10.0):
    """两足间接触力的有界惩罚。"""
    if scale <= 0:
        raise ValueError("Self-collision scale must be positive")
    force = env.scene[sensor_name].data.force
    assert force is not None
    return 1.0 - torch.exp(-force.norm(dim=-1).sum(-1) / scale)


def foot_nonparallel_ground_penalty(env, sensor_name="feet_ground_contact", std=0.25):
    """惩罚接触足底法向与世界向上方向不一致的情况。"""
    if std <= 0:
        raise ValueError("Foot parallelism std must be positive")
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    sites = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)[0]
    quat = robot.data.site_quat_w[:, sites]
    local_normals = torch.zeros_like(quat[..., :3])
    local_normals[..., 2] = 1.0
    normal_w = quat_apply(quat.reshape(-1, 4), local_normals.reshape(-1, 3)).reshape_as(
        local_normals
    )
    alignment = normal_w[..., 2].abs().clamp(0.0, 1.0)
    reward = torch.exp(-((1.0 - alignment) / std).square())
    return ((1.0 - reward) * contact).sum(-1) / contact.sum(-1).clamp_min(1.0)


def sole_tilt_sq(env):
    """各足底相对水平面倾角的 sin^2，形状为 (num_envs, 2)。

    使用足底位点法向 z 分量的平方，因此位点 z 轴指向足内或足外均不影响结果
    （foot_1 指向足外）；与直接使用角度不同，该量在零倾角处平滑。
    """
    robot = env.scene["robot"]
    sites = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)[0]
    return tilt_sq_from_quat(robot.data.site_quat_w[:, sites])


def tilt_sq_from_quat(quat):
    """刚体 z 轴与世界 z 轴夹角的 sin^2（四元数顺序为 w, x, y, z）。"""
    normal_z = 1.0 - 2.0 * (quat[..., 1].square() + quat[..., 2].square())
    return (1.0 - normal_z.square()).clamp_min(0.0)


def _flat_cost(env, sigma):
    """足底完全水平时为 0，饱和值为 1；sigma 约等于代价达到 63% 时的 sin(倾角)。"""
    if sigma <= 0:
        raise ValueError("Flatness sigma must be positive")
    return 1.0 - torch.exp(-sole_tilt_sq(env) / sigma**2)


def stance_flat_penalty(env, sigma=0.05, sensor_name="feet_ground_contact"):
    """惩罚承重足底未平贴地面的情况，取承重足的平均值。"""
    contact = feet_contact(env, sensor_name)
    return (_flat_cost(env, sigma) * contact).sum(-1) / contact.sum(-1).clamp_min(1.0)


def touchdown_flat_penalty(env, sigma=0.05, sensor_name="feet_ground_contact"):
    """足部首次触地时的额外惩罚，要求着地时足底平贴地面。"""
    first = env.scene[sensor_name].compute_first_contact(dt=env.step_dt).float()
    return (_flat_cost(env, sigma) * first).sum(-1)


def pre_landing_flat_penalty(
    env, sigma=0.05, height=0.03, sensor_name="feet_ground_contact"
):
    """惩罚已接近地面但仍未摆平的腾空足底。

    高度采用足底位点相对地面的高度，而非足部刚体原点高度，
    因为实际接触地面的是足底平面。
    """
    robot = env.scene["robot"]
    sites = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)[0]
    sole_height = robot.data.site_pos_w[:, sites, 2] - env.scene.env_origins[:, 2:3]
    approaching = (sole_height < height).float() * (
        1.0 - feet_contact(env, sensor_name)
    )
    return (_flat_cost(env, sigma) * approaching).sum(-1)


def swing_foot_acceleration_penalty(
    env, acceleration_sensors, acceleration_scale, sensor_name="feet_ground_contact"
):
    """仅对腾空足计算世界坐标系绝对加速度的有界代价。"""
    if acceleration_scale <= 0:
        raise ValueError("Acceleration scale must be positive")
    accelerations = torch.stack(
        [env.scene[name].data for name in acceleration_sensors], dim=1
    )
    cost = (accelerations.abs().sum(-1) / acceleration_scale).clamp(max=1.0)
    return (cost * (1.0 - feet_contact(env, sensor_name))).sum(-1)


def swing_foot_excess_height_penalty(
    env,
    max_height,
    height_scale,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
):
    """惩罚腾空足相对地形的离地高度超过允许值。"""
    if max_height < 0 or height_scale <= 0:
        raise ValueError("Height threshold must be non-negative and scale positive")
    heights = env.scene[height_sensor_name].data.heights
    cost = ((heights - max_height).clamp_min(0.0) / height_scale).clamp(max=1.0)
    return (cost * (1.0 - feet_contact(env, sensor_name))).sum(-1)


def joint_acceleration_penalty(env, entity_name="robot"):
    """关节加速度绝对值之和（rad/s²），无需对速度差分。"""
    return env.scene[entity_name].data.joint_acc.abs().sum(dim=-1)


def joint_energy_penalty(env, energy_scale, entity_name="robot"):
    """有界机械功率代价：sum(abs(力矩 * 关节速度))。"""
    if energy_scale <= 0:
        raise ValueError("Energy scale must be positive")
    asset = env.scene[entity_name]
    power = (asset.data.actuator_force * asset.data.joint_vel).abs().sum(-1)
    return (power / energy_scale).clamp(max=1.0)


class SequentialVelocityCommand(UniformVelocityCommandCfg):
    """按环境索引采样前进速度的指令配置。"""

    def build(self, env):
        return _SequentialVelocityCommand(self, env)


class _SequentialVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        values = torch.linspace(
            self.cfg.ranges.lin_vel_x[0],
            self.cfg.ranges.lin_vel_x[1],
            self.num_envs,
            device=self.device,
        )
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = values[env_ids]
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False
        self.is_forward_env[env_ids] = False


def target_speed(env):
    return env.command_manager.get_command("target_speed")[:, :1]


def track_forward_speed(env, std=0.05):
    """高斯跟踪奖励；后退和超速均增大误差。"""
    error = base_lin_vel(env)[:, 0] - target_speed(env)[:, 0]
    return torch.exp(-error.square() / std**2)


def velocity_command_cfg(
    lin_vel_x=(0.0, 0.5),
    lin_vel_y=(0.0, 0.0),
    ang_vel_z=(0.0, 0.0),
    sequential=False,
    cfg_cls=None,
    debug_vis=False,
    **kwargs,
):
    """Zbot 坐标系速度指令：前向、左向、偏航角速度（逆时针为正）。"""
    cfg_cls = cfg_cls or (
        SequentialVelocityCommand if sequential else UniformVelocityCommandCfg
    )
    return cfg_cls(
        entity_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.0,
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=lin_vel_x, lin_vel_y=lin_vel_y, ang_vel_z=ang_vel_z, heading=None
        ),
        debug_vis=debug_vis,
        **kwargs,
    )


def velocity_command(env):
    # 指令项保留 "target_speed" 键，以兼容 play.py 和 sim2sim 的参数覆盖；
    # 此处该键承载三个分量。
    return env.command_manager.get_command("target_speed")


def track_lateral_speed(env, std=0.1):
    error = base_lin_vel(env)[:, 1] - velocity_command(env)[:, 1]
    return torch.exp(-error.square() / std**2)


def track_yaw_rate(env, std=0.25):
    # base_ang_vel 第三轴沿重力向下，因此从上方看逆时针旋转时
    # 该分量为负；见 docs/zbot-coordinate-system.md。
    error = -base_ang_vel(env)[:, 2] - velocity_command(env)[:, 2]
    return torch.exp(-error.square() / std**2)

