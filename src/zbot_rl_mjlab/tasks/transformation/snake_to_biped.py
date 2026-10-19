"""反向任务：从贴地直蛇形升起到站立关节姿态。"""

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import cast

import mujoco
import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from ...robot import DEFAULT_JOINT_POS, JOINT_NAMES, get_spec
from . import mdp
from .actions import JointTargetRateActionCfg
from .env_cfg import TransformationEnvCfg, transformation_scene_cfg
from .poses import SNAKE_JOINT_POS, SNAKE_ROOT_POS, SNAKE_ROOT_QUAT
from .reverse_observations import reserved_hold_observation, support_observation

TASK_ID = "Mjlab-Zbot-6dof-Transform-Snake-To-Biped"
BETA_TASK_ID = TASK_ID + "-Beta"
ARC_TASK_ID = BETA_TASK_ID + "-Arc"


@dataclass
class SnakeToBipedEnvCfg(TransformationEnvCfg):
    support_observations: bool = False
    require_sole_support_for_positive_rewards: bool = True
    initial_joint_pos: tuple[float, ...] = SNAKE_JOINT_POS
    standing_joint_pos: tuple[float, ...] = DEFAULT_JOINT_POS
    joint_standing_error_scale: float = 0.5
    standing_base_height: float = 0.264
    base_height_tolerance: float = 0.02
    sole_height_tolerance: float = 0.005
    sole_alignment_height_std: float = 0.03
    sole_alignment_angle_std_deg: float = 30.0
    sole_angle_tolerance_deg: float = 5.0
    base_z_roll_limit_deg: float = 179.0
    base_gravity_alignment_weight: float = 0.0
    survival_weight: float = 0.0
    curriculum_start_stage: int = 1
    sole_alignment_stage_threshold: float = 1.0
    curriculum_support_hold_s: float = 0.5
    curriculum_support_ready_fraction: float = 0.8
    curriculum_joint_error_threshold: float | None = None
    curriculum_prepare_pose: bool = False
    first_stage_initial_pose_weight: float = -0.02
    first_stage_joint_weight: float = 0.1
    first_stage_prepared_joint_weight: float = 0.0
    standing_stage_joint_weight: float = 1.0
    standing_stage_height_weight: float = 0.0
    standing_stage_velocity_weight: float = 0.0
    standing_stage_ramp_steps: int = 2400
    base_support_distance_weight: float = -0.2
    base_support_distance_std: float = 0.05
    support_angle_limit_deg: float = 25.0
    support_height_limit: float = 0.01
    support_failure_hold_s: float = 0.1
    feet_contact_weight: float = 0.5
    end_joint_flip_weight: float = -1.0
    end_joint_allowed_error: float = 1.0
    first_stage_roll_rate_weight: float = -0.05
    reward_hot_reload_path: str | None = None


def _joint_error(env):
    target = torch.tensor(env.cfg.standing_joint_pos, device=env.device)
    return env.scene["robot"].data.joint_pos - target


def initial_joint_pose_cost(env):
    """相对本实验重置关节姿态的均方位移。"""
    positions = env.scene["robot"].data.joint_pos
    initial = positions.new_tensor(env.cfg.initial_joint_pos)
    cost = (positions - initial).square().mean(-1)
    return torch.nan_to_num(cost, nan=100.0, posinf=100.0).clamp(max=100.0)


def end_joint_flip_cost(env):
    """抑制末端关节绕到 +/-pi 的捷径，不限制中间关节。"""
    if not math.isfinite(env.cfg.end_joint_allowed_error) or env.cfg.end_joint_allowed_error <= 0:
        raise ValueError("end_joint_allowed_error must be finite and positive")
    excess = (_joint_error(env)[:, (0, 5)].abs() - env.cfg.end_joint_allowed_error).clamp_min(0.0)
    return torch.nan_to_num(excess.square().mean(-1), nan=100.0, posinf=100.0)


def axial_roll_rate_cost(env):
    """基座绕纵向局部 Z 轴的实际角速度平方。"""
    robot = env.scene["robot"]
    base_id = robot.find_bodies("base")[0][0]
    velocity = robot.data.body_link_ang_vel_w[:, base_id]
    local_z = torch.zeros_like(velocity)
    local_z[:, 2] = 1.0
    axis = quat_apply(robot.data.body_link_quat_w[:, base_id], local_z)
    rate = (velocity * axis).sum(-1)
    return torch.nan_to_num(rate.square(), nan=100.0, posinf=100.0)


@lru_cache(maxsize=16)
def standing_roll_reference(standing_joint_pos):
    """由原生 FK 推导根部直立站立目标中的基座滚转角。"""
    model = get_spec().compile()
    # 当前安装版本的类型元数据未包含 MuJoCo 的运行时导出。
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, position in zip(JOINT_NAMES, standing_joint_pos, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = position
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    rotation = data.xmat[model.body("base").id].reshape(3, 3)
    return math.atan2(rotation[2, 0], rotation[2, 1])


@lru_cache(maxsize=16)
def standing_gravity_reference(standing_joint_pos):
    """通过 CPU FK 计算根部直立目标中的完整局部重力方向。"""
    model = get_spec().compile()
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, position in zip(JOINT_NAMES, standing_joint_pos, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = position
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    rotation = data.xmat[model.body("base").id].reshape(3, 3)
    return tuple(float(-component) for component in rotation[2])


def base_gravity_alignment_cost(env):
    """完整重力偏差，包括轴向滚转省略的分量。"""
    robot = env.scene["robot"]
    base_id = robot.find_bodies("base")[0][0]
    quaternion = robot.data.body_link_quat_w[:, base_id]
    gravity = torch.zeros_like(quaternion[:, :3])
    gravity[:, 2] = -1.0
    local = quat_apply_inverse(quaternion, gravity)
    norm = local.norm(dim=-1, keepdim=True)
    unit = local / norm.clamp_min(1e-6)
    reference = local.new_tensor(
        standing_gravity_reference(tuple(env.cfg.standing_joint_pos))
    )
    reference = reference / reference.norm().clamp_min(1e-6)
    cost = 1.0 - (unit * reference).sum(-1)
    valid = torch.isfinite(quaternion).all(-1) & (quaternion.norm(dim=-1) > 1e-6)
    valid &= torch.isfinite(local).all(-1) & torch.isfinite(norm).squeeze(-1) & (norm.squeeze(-1) > 1e-6)
    return torch.where(valid, torch.nan_to_num(cost, nan=2.0, posinf=2.0, neginf=2.0).clamp(0.0, 2.0), 2.0)


def axial_roll_angle(base_quat, reference_angle=0.0):
    """基座绕局部 Z 的实际滚转角；零滚转时局部 -Y 朝下。

    ZBOT6.xml 中关节归零的蛇形重置姿态，其重力沿基座 -Y 方向。
    将重力投影到基座 XY 平面可以消除世界偏航，同时保留关节造成的旋转。
    基座 Z 轴竖直时，轴向滚转未定义。
    """
    gravity = torch.zeros_like(base_quat[..., :3])
    gravity[..., 2] = -1.0
    local_gravity = quat_apply_inverse(base_quat, gravity)
    angle = torch.atan2(-local_gravity[..., 0], -local_gravity[..., 1])
    angle = torch.remainder(angle - reference_angle + math.pi, 2.0 * math.pi) - math.pi
    valid = torch.isfinite(base_quat).all(-1)
    valid &= torch.isfinite(local_gravity).all(-1)
    valid &= local_gravity[..., :2].norm(dim=-1) > 1e-4
    return torch.where(valid, angle, torch.full_like(angle, math.pi))


def base_z_roll_angle(env):
    robot = env.scene["robot"]
    base_id = robot.find_bodies("base")[0][0]
    reference = standing_roll_reference(tuple(env.cfg.standing_joint_pos))
    return axial_roll_angle(robot.data.body_link_quat_w[:, base_id], reference)


def _roll_limit(env):
    if not 0.0 < env.cfg.base_z_roll_limit_deg < 180.0:
        raise ValueError("base_z_roll_limit_deg must be between 0 and 180")
    return math.radians(env.cfg.base_z_roll_limit_deg)


def base_z_roll_cost(env):
    return (base_z_roll_angle(env) / _roll_limit(env)).square()


def base_z_roll_exceeded(env):
    return base_z_roll_angle(env).abs() > _roll_limit(env) + 1e-6


def base_z_roll_degrees(env):
    return torch.rad2deg(base_z_roll_angle(env).abs())


def reset_failure_cost(env):
    # TerminationManager 在奖励前运行；即使触发多个安全条件，
    # 其联合掩码也只计费一次。补偿 RewardManager 的 dt 缩放。
    return env.termination_manager.terminated.float() / env.step_dt


def survival_reward(env):
    """正常控制步的恒定偏移；真正失败时不给生存奖励。"""
    return (~env.termination_manager.terminated).float()


def joint_standing_reward(env):
    """负的 Smooth L1 姿态误差，为远离目标的关节提供线性反馈。

    正课程权重用于缩放此惩罚。支撑丢失时不屏蔽该负反馈；
    足部支撑保护仍单独执行。
    """
    scale = env.cfg.joint_standing_error_scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("joint_standing_error_scale must be finite and positive")
    error = _joint_error(env).abs()
    cost = torch.where(error < scale, 0.5 * error.square() / scale, error - 0.5 * scale)
    return -torch.nan_to_num(cost, nan=100.0, posinf=100.0).mean(-1)


def base_height(env):
    """基座坐标原点相对任务平地的高度。"""
    data, base_id = mdp._base(env)
    return data.body_link_pos_w[:, base_id, 2] - env.scene.env_origins[:, 2]


def base_height_progress_reward(env):
    """基座原点的高度进展，受第二阶段足部支撑保护约束。"""
    initial_height = SNAKE_ROOT_POS[2]
    span = env.cfg.standing_base_height - initial_height
    if span <= 0:
        raise ValueError("standing_base_height must exceed the initial snake height")
    progress = (base_height(env) - initial_height) / span
    return constrain_standing_reward(env, torch.nan_to_num(progress, nan=-1.0).clamp(-1.0, 1.0))


def sole_height_cost(env):
    """两个足底位点到任务平地的绝对距离平均值。"""
    if env.cfg.sole_alignment_height_std <= 0:
        raise ValueError("sole_alignment_height_std must be positive")
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    sole_height = robot.data.site_pos_w[:, ids, 2] - env.scene.env_origins[:, 2, None]
    cost = sole_height.abs() / env.cfg.sole_alignment_height_std
    return torch.nan_to_num(cost, nan=4.0, posinf=4.0).clamp(max=4.0).mean(-1)


def base_support_distance(env):
    """基座坐标原点到足底线段的水平距离。"""
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    base_id = robot.find_bodies("base")[0][0]
    base_xy = robot.data.body_link_pos_w[:, base_id, :2]
    feet = robot.data.site_pos_w[:, ids, :2]
    start = feet[:, 0]
    segment = feet[:, 1] - start
    length_sq = segment.square().sum(-1)
    fraction = ((base_xy - start) * segment).sum(-1) / length_sq.clamp_min(1e-12)
    fraction = fraction.clamp(0.0, 1.0)
    closest = start + fraction[..., None] * segment
    return (base_xy - closest).norm(dim=-1)


def base_support_distance_cost(env):
    """归一化支撑线段距离代价；应使用负奖励权重。"""
    if not math.isfinite(env.cfg.base_support_distance_std) or env.cfg.base_support_distance_std <= 0:
        raise ValueError("base_support_distance_std must be finite and positive")
    distance = base_support_distance(env)
    cost = distance / env.cfg.base_support_distance_std
    return torch.nan_to_num(cost, nan=4.0, posinf=4.0).clamp(max=4.0)


def _sole_direction_cosine(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    quat = robot.data.site_quat_w[:, ids]
    normal_z = 1.0 - 2.0 * (quat[..., 1].square() + quat[..., 2].square())
    # foot_0 的物理足底向外法向为 -site_Z；foot_1 为 +site_Z。
    cosine = (normal_z * quat.new_tensor([1.0, -1.0])).clamp(-1.0, 1.0)
    return torch.where(torch.isfinite(quat).all(-1), cosine, float("nan"))


def _sole_ground_alignment(env, angle_power: float = 2.0):
    """各足有符号足底方向得分，与足部高度无关。"""
    if env.cfg.sole_alignment_angle_std_deg <= 0:
        raise ValueError("sole_alignment_angle_std_deg must be positive")
    if not math.isfinite(angle_power) or angle_power <= 0:
        raise ValueError("angle_power must be finite and positive")
    cosine = _sole_direction_cosine(env)
    angle = torch.acos(cosine)
    sigma = math.radians(env.cfg.sole_alignment_angle_std_deg)
    # 指数 1 在足底水平处保留非零单侧斜率；指数 2 为旧版行为。
    score = 0.5 * (cosine - 1.0) + torch.exp(-(angle / sigma).pow(angle_power))
    return torch.nan_to_num(score, nan=0.0)


def sole_ground_alignment_reward(env, angle_power: float = 2.0):
    # 保留双足反馈，同时提高较差一足的影响。
    scores = _sole_ground_alignment(env, angle_power=angle_power)
    return 0.5 * (scores.mean(-1) + scores.amin(-1))


def sole_angles_degrees(env):
    """各足相对朝下方向的夹角；无效方向记为 180 度。"""
    cosine = _sole_direction_cosine(env)
    return torch.nan_to_num(torch.rad2deg(torch.acos(cosine)), nan=180.0)


def sole_angle_degrees(env, foot_index):
    return sole_angles_degrees(env)[:, foot_index]


def both_soles_within_10deg(env):
    return (sole_angles_degrees(env).amax(-1) < 10.0).float()


def sole_support_valid(env):
    """两个足底位点均须朝下且靠近地面。"""
    cfg = env.cfg
    if not math.isfinite(cfg.support_angle_limit_deg) or not 0 < cfg.support_angle_limit_deg < 90:
        raise ValueError("support_angle_limit_deg must be between 0 and 90")
    if not math.isfinite(cfg.support_height_limit) or cfg.support_height_limit <= 0:
        raise ValueError("support_height_limit must be finite and positive")
    robot = env.scene["robot"]
    ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    heights = robot.data.site_pos_w[:, ids, 2] - env.scene.env_origins[:, 2, None]
    cosine = _sole_direction_cosine(env)
    return (
        both_feet_ground_contact(env)
        & torch.isfinite(cosine).all(-1)
        & torch.isfinite(heights).all(-1)
        & (cosine.amin(-1) >= math.cos(math.radians(cfg.support_angle_limit_deg)))
        & (heights.abs().amax(-1) <= cfg.support_height_limit)
    )


def both_feet_ground_contact(env):
    """分别检测两足刚体与地形的实际接触。"""
    found = env.scene["feet_ground_contact"].data.found
    if found is None:
        raise ValueError("feet_ground_contact sensor must provide found data")
    return (found > 0).all(-1)


def both_feet_contact_metric(env):
    return both_feet_ground_contact(env).float()


def constrain_standing_reward(env, reward):
    if not env.cfg.require_sole_support_for_positive_rewards:
        return reward
    stage = getattr(env, "_standing_curriculum_stage", env.cfg.curriculum_start_stage)
    if stage == 1:
        return reward
    # 支撑丢失时保留全部负奖励项；仅门控正向增益。
    return torch.where(sole_support_valid(env), reward, reward.clamp(max=0.0))


@dataclass
class SupportProtectionState:
    support_armed: torch.Tensor
    support_violation_s: torch.Tensor
    support_hold_s: torch.Tensor
    last_episode_ready: torch.Tensor


def support_protection_state(env):
    if not hasattr(env, "_support_protection_state"):
        env._support_protection_state = SupportProtectionState(
            support_armed=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            support_violation_s=torch.zeros(env.num_envs, device=env.device),
            support_hold_s=torch.zeros(env.num_envs, device=env.device),
            last_episode_ready=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        )
    return env._support_protection_state


def reset_reverse_transition(env, env_ids):
    mdp.reset_transformation(env, env_ids)
    task = support_protection_state(env)
    ids = slice(None) if env_ids is None else env_ids
    task.support_armed[ids] = False
    task.support_violation_s[ids] = 0.0
    task.support_hold_s[ids] = 0.0


def update_support_protection(env, fresh):
    stage = getattr(env, "_standing_curriculum_stage", env.cfg.curriculum_start_stage)
    task = support_protection_state(env)
    valid = sole_support_valid(env)
    held = torch.where(valid, task.support_hold_s + env.step_dt, 0.0)
    task.support_hold_s.copy_(torch.where(fresh, held, task.support_hold_s))
    if stage == 1:
        return
    armed = task.support_armed | valid
    elapsed = torch.where(armed & ~valid, task.support_violation_s + env.step_dt, 0.0)
    task.support_armed.copy_(torch.where(fresh, armed, task.support_armed))
    task.support_violation_s.copy_(torch.where(fresh, elapsed, task.support_violation_s))


def foot_support_lost(env):
    if not math.isfinite(env.cfg.support_failure_hold_s) or env.cfg.support_failure_hold_s <= 0:
        raise ValueError("support_failure_hold_s must be finite and positive")
    stage = getattr(env, "_standing_curriculum_stage", env.cfg.curriculum_start_stage)
    task = support_protection_state(env)
    return (
        (stage == 2)
        & task.support_armed
        & (task.support_violation_s >= env.cfg.support_failure_hold_s - 1e-6)
    )


def sole_support_valid_metric(env):
    return sole_support_valid(env).float()


def base_vertical_velocity(env):
    """基座坐标原点沿世界 Z 轴的线速度，并非其质心速度。"""
    data, base_id = mdp._base(env)
    return data.body_link_lin_vel_w[:, base_id, 2]


def base_ascent_velocity_reward(env, target_speed=0.03, pose_error_range: tuple[float, float] | None = None):
    if target_speed <= 0:
        raise ValueError("target_speed must be positive")
    velocity = base_vertical_velocity(env)
    # 向上进展在目标速度处饱和，不惩罚超速。
    score = 1.0 - ((velocity / target_speed).clamp(max=1.0) - 1.0).square()
    score = torch.where(torch.isfinite(velocity), score, -4.0)
    score = torch.nan_to_num(score, nan=-4.0, posinf=-4.0, neginf=-4.0).clamp(-4.0, 1.0)
    if pose_error_range is not None:
        zero_error, full_error = pose_error_range
        if not (math.isfinite(zero_error) and math.isfinite(full_error) and 0 <= zero_error < full_error):
            raise ValueError("pose_error_range must satisfy 0 <= zero < full, both finite")
        error = torch.nan_to_num(_joint_error(env).abs(), nan=math.inf).amax(-1)
        fraction = ((error - zero_error) / (full_error - zero_error)).clamp(0.0, 1.0)
        # Smoothstep 在接近目标姿态时同时消除上升增益和下降代价。
        score = score * fraction.square() * (3.0 - 2.0 * fraction)
    return constrain_standing_reward(env, score)


def final_condition(env, joint_tolerance=None):
    tolerance = env.cfg.joint_tolerance if joint_tolerance is None else joint_tolerance
    robot = env.scene["robot"]
    data = robot.data
    base_id = robot.find_bodies("base")[0][0]
    site_ids, _ = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)
    soles = data.site_pos_w[:, site_ids, 2] - env.scene.env_origins[:, 2, None]
    return (
        (_joint_error(env).abs().amax(-1) < tolerance)
        & both_feet_ground_contact(env)
        & torch.isfinite(data.body_link_pos_w[:, base_id]).all(-1)
        & (soles.abs().amax(-1) < env.cfg.sole_height_tolerance)
        & (
            _sole_direction_cosine(env).amin(-1)
            > math.cos(math.radians(env.cfg.sole_angle_tolerance_deg))
        )
        & ~base_z_roll_exceeded(env)
        & (data.joint_vel.square().mean(-1).sqrt() < env.cfg.joint_speed_tolerance)
        & (
            data.body_link_lin_vel_w[:, base_id].norm(dim=-1)
            < env.cfg.base_speed_tolerance
        )
    )


def advance(env):
    if hasattr(env, "_standing_curriculum_stage"):
        apply_standing_stage_weights(env)
    task = mdp.state(env)
    fresh = task.last_step != env.common_step_counter
    update_support_protection(env, fresh)
    task.advance_hold(final_condition(env), env.common_step_counter, env.step_dt)


def update_state(env):
    advance(env)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def success(env):
    return (
        mdp.state(env).hold_time >= env.cfg.success_hold_s - 1e-5
    ) & final_condition(env)


def joint_standing_error_metric(env):
    return _joint_error(env).abs().mean(-1)


def apply_standing_stage_weights(env):
    """阶段提升后，每个控制步渐变调整实际奖励权重。"""
    cfg = env.cfg
    if cfg.standing_stage_ramp_steps < 0:
        raise ValueError("standing_stage_ramp_steps must be nonnegative")
    if not math.isfinite(cfg.first_stage_prepared_joint_weight) or cfg.first_stage_prepared_joint_weight < 0:
        raise ValueError("first_stage_prepared_joint_weight must be finite and nonnegative")
    stage = env._standing_curriculum_stage
    scale = 0.0
    if stage == 2:
        if cfg.standing_stage_ramp_steps == 0:
            scale = 1.0
        else:
            elapsed = env.common_step_counter - env._standing_stage_start_step
            scale = min(max(elapsed / cfg.standing_stage_ramp_steps, 0.0), 1.0)
    for name, weight in (
        ("base_height_progress", cfg.standing_stage_height_weight),
        ("base_ascent_velocity", cfg.standing_stage_velocity_weight),
    ):
        env.reward_manager.get_term_cfg(name).weight = weight * scale
    starting_joint_weight = cfg.first_stage_joint_weight
    if cfg.curriculum_prepare_pose and (
        stage == 2 or getattr(env, "_standing_pose_prepared", False)
    ):
        starting_joint_weight = cfg.first_stage_prepared_joint_weight
    env.reward_manager.get_term_cfg("joint_standing").weight = (
        starting_joint_weight
        + (cfg.standing_stage_joint_weight - starting_joint_weight) * scale
    )
    env.reward_manager.get_term_cfg("base_support_distance").weight = cfg.base_support_distance_weight
    env.reward_manager.get_term_cfg("base_gravity_alignment").weight = cfg.base_gravity_alignment_weight
    env.reward_manager.get_term_cfg("survival").weight = cfg.survival_weight
    env.reward_manager.get_term_cfg("initial_joint_pose").weight = (
        cfg.first_stage_initial_pose_weight if stage == 1 else 0.0
    )
    env.reward_manager.get_term_cfg("end_joint_flip").weight = cfg.end_joint_flip_weight
    env.reward_manager.get_term_cfg("axial_roll_rate").weight = (
        cfg.first_stage_roll_rate_weight if stage == 1 else 0.0
    )
    return scale


def alignment_to_standing_curriculum(env, env_ids):
    """已结束回合达到对齐门槛后，将整次训练提升到下一阶段。

    在 reset 清除 RewardManager 按 dt 积分的回合累计量之前调用。
    门槛严格采用 Episode_Reward 日志的归一化方式，而非未加权得分
    （其不超过 1）或按 dt 缩放后的单步奖励。第二阶段在后续所有回合重置中持续生效。
    """
    cfg = env.cfg
    if cfg.curriculum_start_stage not in (1, 2):
        raise ValueError("curriculum_start_stage must be 1 or 2")
    if not math.isfinite(cfg.curriculum_support_hold_s) or cfg.curriculum_support_hold_s <= 0:
        raise ValueError("curriculum_support_hold_s must be finite and positive")
    if not 0 < cfg.curriculum_support_ready_fraction <= 1:
        raise ValueError("curriculum_support_ready_fraction must be in (0, 1]")
    pose_threshold = cfg.curriculum_joint_error_threshold
    if pose_threshold is not None and (
        isinstance(pose_threshold, bool)
        or not math.isfinite(pose_threshold)
        or pose_threshold <= 0
    ):
        raise ValueError("curriculum_joint_error_threshold must be finite and positive or None")
    alignment_weight = env.reward_manager.get_term_cfg("sole_ground_alignment").weight
    if not 0.0 <= cfg.sole_alignment_stage_threshold < alignment_weight:
        raise ValueError("sole alignment stage threshold must be reachable with its reward weight")
    stage = getattr(env, "_standing_curriculum_stage", cfg.curriculum_start_stage)
    ids = slice(None) if env_ids is None else env_ids
    completed = env.episode_length_buf[ids] > 0
    # RewardManager 未提供回合累计量的公共访问接口；将此依赖
    # 限制在局部，并在管理器 reset() 清除之前读取。
    scores = env.reward_manager._episode_sums["sole_ground_alignment"][ids]
    scores = scores[completed] / env.max_episode_length_s
    alignment = scores.mean().item() if scores.numel() else 0.0
    support = support_protection_state(env)
    held = support.support_hold_s[ids]
    failed = env.termination_manager.terminated[ids]
    ready = (held >= cfg.curriculum_support_hold_s - 1e-6) & ~failed
    if pose_threshold is not None:
        ready &= _joint_error(env)[ids].abs().amax(-1) < pose_threshold
    support.last_episode_ready[ids] = torch.where(completed, ready, support.last_episode_ready[ids])
    # 保留每个环境最近已结束回合的结果。异步重置的小批次
    # 不应仅凭一次幸运回合就提升所有环境的阶段。
    ready_fraction = support.last_episode_ready.float().mean().item()
    alignment_wait_steps = 0
    changed_step = getattr(env, "_standing_alignment_hot_changed_step", None)
    if changed_step is not None:
        alignment_wait_steps = max(
            0, env.max_episode_length - (env.common_step_counter - changed_step)
        )
    if cfg.curriculum_prepare_pose and (
        stage == 2 or (alignment > cfg.sole_alignment_stage_threshold and alignment_wait_steps == 0)
    ):
        env._standing_pose_prepared = True
    if (stage == 1 and alignment > cfg.sole_alignment_stage_threshold
            and ready_fraction >= cfg.curriculum_support_ready_fraction
            and alignment_wait_steps == 0):
        stage = 2
        env._standing_stage_start_step = env.common_step_counter
    elif stage == 2 and not hasattr(env, "_standing_stage_start_step"):
        # 显式从第二阶段开始（包括回放）时使用完整目标。
        env._standing_stage_start_step = env.common_step_counter - cfg.standing_stage_ramp_steps
    env._standing_curriculum_stage = stage
    ramp = apply_standing_stage_weights(env)
    return {"stage": float(stage), "alignment": alignment, "ramp": ramp,
            "support_ready_fraction": ready_fraction,
            "joint_error_threshold": -1.0 if pose_threshold is None else float(pose_threshold),
            "pose_prepared": float(getattr(env, "_standing_pose_prepared", False)),
            "alignment_hot_wait_steps": float(alignment_wait_steps)}


def snake_to_biped_env_cfg(play=False, cfg_cls=SnakeToBipedEnvCfg, action_cfg_cls=JointTargetRateActionCfg, compact=False):
    """直接创建最终起立配置类型，再装配方向相关的行为。"""
    cfg = cast(SnakeToBipedEnvCfg, transformation_scene_cfg(
        play=play, cfg_cls=cfg_cls, action_cfg_cls=action_cfg_cls,
    ))
    cfg.metrics = {
        "joint_error": MetricsTermCfg(func=joint_standing_error_metric),
        "com_height": MetricsTermCfg(func=mdp.com_height),
        "success": MetricsTermCfg(func=success, reduce="last"),
    }
    # 回放评估完整站立目标；训练从第一阶段开始。
    cfg.curriculum_start_stage = 2 if play else 1
    for group in cfg.observations.values():
        group.terms["hold_time"] = ObservationTermCfg(func=reserved_hold_observation)
        group.terms["support_state"] = ObservationTermCfg(func=support_observation)
    standing = cfg.curriculum_start_stage == 2
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
        ),
    )
    robot = cfg.scene.entities["robot"]
    robot.init_state = replace(
        robot.init_state,
        pos=SNAKE_ROOT_POS,
        rot=SNAKE_ROOT_QUAT,
        joint_pos=dict(zip(JOINT_NAMES, cfg.initial_joint_pos, strict=True)),
        joint_vel={".*": 0.0},
    )
    action = cfg.actions["joint_pos"]
    if not isinstance(action, JointTargetRateActionCfg):
        raise TypeError("reverse transformation requires rate-controlled joint targets")
    action.joint_target_bounds = {
        JOINT_NAMES[i]: (cfg.standing_joint_pos[i] - cfg.end_joint_allowed_error,
                         cfg.standing_joint_pos[i] + cfg.end_joint_allowed_error)
        for i in (0, 5)
    }
    if compact:
        return cfg
    cfg.rewards = {
        "end_joint_flip": RewardTermCfg(
            func=end_joint_flip_cost,
            weight=cfg.end_joint_flip_weight,
        ),
        "axial_roll_rate": RewardTermCfg(
            func=axial_roll_rate_cost,
            weight=0.0 if standing else cfg.first_stage_roll_rate_weight,
        ),
        "initial_joint_pose": RewardTermCfg(
            func=initial_joint_pose_cost,
            weight=0.0 if standing else cfg.first_stage_initial_pose_weight,
        ),
        "joint_standing": RewardTermCfg(
            func=joint_standing_reward,
            weight=cfg.standing_stage_joint_weight if standing else cfg.first_stage_joint_weight,
        ),
        "base_height_progress": RewardTermCfg(
            func=base_height_progress_reward,
            weight=cfg.standing_stage_height_weight if standing else 0.0,
        ),
        "base_ascent_velocity": RewardTermCfg(
            func=base_ascent_velocity_reward,
            weight=cfg.standing_stage_velocity_weight if standing else 0.0,
            params={"target_speed": 0.1},
        ),
        "base_support_distance": RewardTermCfg(
            func=base_support_distance_cost,
            weight=cfg.base_support_distance_weight,
        ),
        "joint_symmetry": RewardTermCfg(
            func=mdp.joint_symmetry_cost,
            weight=-1.0
        ),
        "sole_ground_alignment": RewardTermCfg(
            func=sole_ground_alignment_reward,
            weight=2.0,
        ),
        "both_feet_contact": RewardTermCfg(func=both_feet_contact_metric, weight=cfg.feet_contact_weight),
        "sole_height": RewardTermCfg(func=sole_height_cost, weight=-1.0),
        "base_z_roll": RewardTermCfg(func=base_z_roll_cost, weight=-1.0),
        "base_gravity_alignment": RewardTermCfg(func=base_gravity_alignment_cost, weight=cfg.base_gravity_alignment_weight),
        "reset_failure": RewardTermCfg(func=reset_failure_cost, weight=-10.0),
        "survival": RewardTermCfg(func=survival_reward, weight=cfg.survival_weight),
    }
    cfg.terminations["update_state"] = TerminationTermCfg(func=update_state)
    cfg.terminations["base_z_roll"] = TerminationTermCfg(func=base_z_roll_exceeded)
    cfg.terminations["foot_support_lost"] = TerminationTermCfg(func=foot_support_lost)
    cfg.events["reset_transition"].func = reset_reverse_transition
    cfg.curriculum = {
        "alignment_to_standing": CurriculumTermCfg(func=alignment_to_standing_curriculum)
    }
    cfg.metrics["joint_error"] = MetricsTermCfg(func=joint_standing_error_metric)
    cfg.metrics["base_height"] = MetricsTermCfg(func=base_height)
    cfg.metrics["base_vertical_velocity"] = MetricsTermCfg(func=base_vertical_velocity)
    cfg.metrics["base_support_distance"] = MetricsTermCfg(func=base_support_distance)
    cfg.metrics["sole_support_valid"] = MetricsTermCfg(func=sole_support_valid_metric)
    cfg.metrics["both_feet_contact"] = MetricsTermCfg(func=both_feet_contact_metric)
    cfg.metrics["success"] = MetricsTermCfg(func=success, reduce="last")
    cfg.metrics["base_z_roll_deg"] = MetricsTermCfg(
        func=base_z_roll_degrees, reduce="max"
    )
    for foot in (0, 1):
        cfg.metrics[f"sole_{foot}_angle_deg"] = MetricsTermCfg(
            func=sole_angle_degrees, params={"foot_index": foot}, reduce="mean"
        )
    cfg.metrics["both_soles_within_10deg"] = MetricsTermCfg(
        func=both_soles_within_10deg, reduce="mean"
    )
    return cfg
