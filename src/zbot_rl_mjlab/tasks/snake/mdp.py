"""蛇形教师的头部坐标、整机推进、柔和接触和安全边界。"""

import math
from typing import cast

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul

from ..transformation.mdp import body_masses, com_state
from .commands import SnakeVelocityCommand

ACTOR_DIM = 40
CRITIC_DIM = 55
LINEAR_TRACKING_STD = .05  # m/s，匹配Zbot厘米每秒量级的平移速度。
YAW_TRACKING_STD = .3  # rad/s。


def command_term(env) -> SnakeVelocityCommand:
    return cast(SnakeVelocityCommand, env.command_manager.get_term("motion"))


def head_index(env):
    return env.scene["robot"].find_bodies(env.cfg.head_body)[0][0]


def motion_site_index(env):
    return env.scene["robot"].find_sites(env.cfg.motion_site)[0][0]


def head_forward(env):
    """新任务由运动site的Z和重力确定前向，旧检查点保留头部朝外轴。"""
    if env.cfg.motion_frame == 'body_gravity':
        return body_heading(env)
    if env.cfg.motion_frame != 'legacy_head':
        raise ValueError('未知蛇形运动坐标系')
    data = env.scene["robot"].data
    quat = data.body_link_quat_w[:, head_index(env)]
    axis = quat.new_tensor(env.cfg.head_forward_axis).expand(env.num_envs, -1)
    forward = quat_apply(quat, axis)[:, :2]
    # 摄像头指向竖直时航向本身退化，此时依靠倾斜惩罚退出该姿态。
    backup = quat_apply(quat, quat.new_tensor((1., 0., 0.)).expand_as(axis))[:, :2]
    forward = torch.where((forward.norm(dim=-1) > .1)[:, None], forward, backup)
    return forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _body_motion_forward(quat):
    """蛇形先由重力叉乘模块Z确定左向，再由左向叉乘向上确定前向。"""
    up = quat.new_tensor((0., 0., 1.)).expand(*quat.shape[:-1], 3)
    body_z = quat_apply(quat, up)
    left = torch.linalg.cross(-up, body_z)
    norm = left.norm(dim=-1, keepdim=True)
    left = left / norm.clamp_min(1e-6)
    forward = torch.linalg.cross(left, up)
    valid = torch.isfinite(forward).all(-1) & (norm[:, 0] > 1e-4)
    return forward[:, :2], valid


def body_heading(env):
    """使用运动site的前向；site Z竖直退化时保留最近有效方向。"""
    robot = env.scene['robot']
    quat = robot.data.site_quat_w[:, motion_site_index(env)]
    forward, valid = _body_motion_forward(quat)
    z = quat.new_tensor((0., 0., 1.)).expand(env.num_envs, -1)
    if not hasattr(env, '_snake_forward_w'):
        # 尚无有效历史且Z竖直时，从身体X轴初始化，不引入固定世界航向。
        x = quat.new_tensor((1., 0., 0.)).expand_as(z)
        fallback = quat_apply(quat, x)[:, :2]
        env._snake_forward_w = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    env._snake_forward_w.copy_(torch.where(valid[:, None], forward, env._snake_forward_w))
    return env._snake_forward_w


def motion_axes(env):
    """蛇形前向、左向、向上组成右手系；偏航取世界+Z为正。"""
    forward = head_forward(env)
    left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    return forward, left


def head_gravity(env):
    data = env.scene["robot"].data
    quat = data.body_link_quat_w[:, head_index(env)]
    gravity = quat.new_tensor((0., 0., -1.)).expand(env.num_envs, -1)
    return quat_apply_inverse(quat, gravity)


def head_tilt(env):
    target = head_gravity(env).new_tensor(env.cfg.head_up_axis)
    cosine = -(head_gravity(env) * target).sum(-1)
    return cosine.clamp(-1., 1.).acos()


def head_tilt_limit_reached(env):
    """蛇头总倾角达到阈值即终止；微小容差覆盖四元数的浮点舍入。"""
    return head_tilt(env) >= env.cfg.head_tilt_limit - 1e-6


def head_tilt_reset_cost(env):
    """仅终止帧计费，抵消奖励管理器的dt缩放后成为一次性事件成本。"""
    return env.termination_manager.get_term('head_tilt').float() / env.step_dt


def head_height(env):
    return env.scene["robot"].data.body_link_pos_w[:, head_index(env), 2] - env.scene.env_origins[:, 2]


def world_com_velocity(env):
    """全身质量加权质心的世界XYZ速度；不以头部摆动代替整机推进。"""
    velocity = env.scene["robot"].data.body_com_lin_vel_w
    mass = body_masses(env)
    return (velocity * mass[..., None]).sum(-2) / mass.sum(-1, keepdim=True).clamp_min(1e-6)


def forward_speed(env):
    """整机质心沿蛇形前向的带符号速度；倒退得负分。"""
    forward, _ = motion_axes(env)
    return (world_com_velocity(env)[:, :2] * forward).sum(-1)


def lateral_speed_cost(env):
    """整机质心沿蛇形左向的速度绝对值。"""
    _, left = motion_axes(env)
    return (world_com_velocity(env)[:, :2] * left).sum(-1).abs()


def vertical_speed_cost(env):
    """蛇形向上轴与重力反向，上下运动均计费。"""
    return world_com_velocity(env)[:, 2].abs()


def world_velocity_component(env, axis: int):
    return world_com_velocity(env)[:, axis]


def observation_reference(env):
    """世界+X在foot_0刚体坐标中的单位向量，提供随机初始航向的可观测性。"""
    reference = env.cfg.observation_reference
    if reference == 'command':
        return command_term(env).command
    if reference != 'world_x':
        raise ValueError('未知蛇形观测参考系')
    quat = env.scene["robot"].data.body_link_quat_w[:, head_index(env)]
    world_x = quat.new_tensor((1., 0., 0.)).expand(env.num_envs, -1)
    return quat_apply_inverse(quat, world_x)


def planar_velocity(env):
    """平移使用全机质心速度，摆动头部或中间模块不能冒充推进。"""
    data = env.scene["robot"].data
    linear = world_com_velocity(env)[:, :2]
    forward, left = motion_axes(env)
    yaw = (data.body_link_ang_vel_w[:, head_index(env), 2]
           if env.cfg.motion_frame == 'legacy_head'
           else data.site_ang_vel_w[:, motion_site_index(env), 2])
    return torch.stack(((linear * forward).sum(-1), (linear * left).sum(-1), yaw), -1)


def tracking_components(actual, requested):
    """旧诊断兼容接口；当前训练使用MJLab指数速度跟踪奖励。"""
    tolerance = actual.new_tensor((.04, .04, .20))
    scale = torch.maximum(requested.abs(), tolerance)
    # 平方误差相对静止基线的改善：零均值往返不能靠瞬时命中刷正分。
    progress = (2 * actual * requested - actual.square()) / scale.square()
    active = requested.abs() > 1e-3
    count = active.sum(-1).clamp_min(1)
    translation = (progress[:, :2] * active[:, :2]).sum(-1) / count
    yaw = progress[:, 2] * active[:, 2] / count
    drift = torch.where(active, 0., (actual / tolerance).square()).sum(-1)
    drift = torch.where(active.any(-1), drift.clamp(max=20), 0.)
    return translation, yaw, drift


def tracking_score(actual, requested):
    """保留原合计接口；真正静止时不获得正的跟踪奖励。"""
    translation, yaw, drift = tracking_components(actual, requested)
    return translation + yaw - .05 * drift


def _tracking_components(env):
    translation, yaw, drift = tracking_components(planar_velocity(env), command_term(env).command)
    # 接地门控只看运动跟踪；未启用的漂移成本不得影响奖励门控。
    total = translation + yaw
    grounded = ground_support(env) > .15
    # 共同门控保持拆分前的合计；离地时正负分项抵消后的总额也不能为正。
    gate = (grounded | (total <= 0)).float()
    return translation * gate, yaw * gate, drift * gate


def velocity_tracking(env):
    """仅供旧诊断脚本使用；当前训练配置不调用此旧公式。"""
    translation, yaw, drift = _tracking_components(env)
    return translation + yaw - .05 * drift


def translation_tracking(env, std: float = LINEAR_TRACKING_STD):
    """参考MJLab/G1：跟踪site运动坐标中的质心速度，竖直目标为零。"""
    actual = planar_velocity(env)
    target = command_term(env).command
    xy_error = (target[:, :2] - actual[:, :2]).square().sum(-1)
    z_error = com_state(env)[:, 1].square()
    return torch.exp(-(xy_error + z_error) / std**2)


def yaw_tracking(env, std: float = YAW_TRACKING_STD):
    """参考MJLab/G1：跟踪竖直轴角速度，非yaw角速度目标为零。"""
    actual = planar_velocity(env)
    target = command_term(env).command
    data = env.scene['robot'].data
    angular = (data.body_link_ang_vel_w[:, head_index(env)]
               if env.cfg.motion_frame == 'legacy_head'
               else data.site_ang_vel_w[:, motion_site_index(env)])
    z_error = (target[:, 2] - actual[:, 2]).square()
    xy_error = angular[:, :2].square().sum(-1)
    return torch.exp(-(z_error + xy_error) / std**2)


def uncommanded_motion_cost(env):
    return _tracking_components(env)[2]


def target_windup_cost(env):
    """只惩罚已顶限却仍向外推的原始速度请求，保留范围内的自由动作。"""
    target = env.action_manager.get_term('joint_pos').commanded_target
    limit = env.cfg.actions['joint_pos'].target_limit
    at_limit = target.abs() >= limit - 1e-4
    # 使用执行前原始输出，避免输出已超过±1后裁剪抹掉代价差异。
    outward = (env.action_manager.action * target.sign()).clamp_min(0.)
    return (outward.square() * at_limit).mean(-1)


class TranslationProgressState:
    """对稳定平移请求的持续进度滤波；重复读取不重复推进时间。"""

    def __init__(self, num_envs, device, cfg):
        for value in (cfg.stall_min_speed, cfg.stall_grace_s, cfg.stall_filter_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("停滞速度、宽限时间和滤波时间必须为正有限值")
        if not math.isfinite(cfg.stall_progress_fraction) or not 0 < cfg.stall_progress_fraction <= 1:
            raise ValueError("停滞进度比例必须在(0, 1]内")
        self.cfg = cfg
        self.request = torch.full((num_envs, 3), float('nan'), device=device)
        self.age = torch.zeros(num_envs, device=device)
        self.progress = torch.zeros_like(self.age)
        self.cost = torch.zeros_like(self.age)
        self.last_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.request[ids] = float('nan')
        self.age[ids] = 0
        self.progress[ids] = 0
        self.cost[ids] = 0
        self.last_step[ids] = -1

    def update(self, actual, command, requested, step, dt):
        fresh = self.last_step != step
        changed = ~torch.isfinite(self.request).all(-1) | ((requested - self.request).abs().amax(-1) > 1e-4)
        speed = command[:, :2].norm(dim=-1)
        requested_speed = requested[:, :2].norm(dim=-1)
        active = (requested_speed >= self.cfg.stall_min_speed) & (speed >= self.cfg.stall_min_speed)
        # 保留负进度；单次高速脉冲不能长期遮蔽随后停滞。
        # 使用原请求方向，切到侧移后残留的前进惯性不能冒充侧移推进。
        ratio = (actual[:, :2] * requested[:, :2]).sum(-1) / requested_speed.square().clamp_min(1e-8)
        ratio = torch.nan_to_num(ratio, nan=-1., posinf=1., neginf=-1.).clamp(-1, 1)
        alpha = 1 - math.exp(-dt / self.cfg.stall_filter_s)
        cleared = changed | ~active
        age = torch.where(cleared, 0., self.age + dt)
        progress = torch.where(cleared, 0., self.progress + alpha * (ratio - self.progress))
        cost = ((self.cfg.stall_progress_fraction - progress) / self.cfg.stall_progress_fraction).clamp(0, 1)
        cost = torch.where(active & (age >= self.cfg.stall_grace_s - 1e-6), cost, 0.)
        self.age.copy_(torch.where(fresh, age, self.age))
        self.progress.copy_(torch.where(fresh, progress, self.progress))
        self.cost.copy_(torch.where(fresh, cost, self.cost))
        self.request.copy_(torch.where(fresh[:, None], requested, self.request))
        self.last_step[fresh] = step
        return self.cost


def translation_progress_state(env):
    if not hasattr(env, '_snake_translation_progress'):
        env._snake_translation_progress = TranslationProgressState(env.num_envs, env.device, env.cfg)
    return env._snake_translation_progress


def translation_stall_cost(env):
    term = command_term(env)
    return translation_progress_state(env).update(
        planar_velocity(env), term.command, term.requested_velocity,
        env.common_step_counter, env.step_dt,
    )


def translation_progress_ratio(env):
    translation_stall_cost(env)
    return translation_progress_state(env).progress


def translation_stall_fraction(env):
    return (translation_stall_cost(env) > 0).float()


def translation_command_fraction(env):
    term = command_term(env)
    return ((term.requested_velocity[:, :2].norm(dim=-1) >= env.cfg.stall_min_speed)
            & (term.command[:, :2].norm(dim=-1) >= env.cfg.stall_min_speed)).float()


def low_posture(env):
    return torch.exp(-((com_state(env)[:, 0] - .055) / .055).square())


def joint_pose_deviation_cost(env):
    """相对初始关节角的均方偏离，允许适度弯曲，持续惩罚大幅卷曲。"""
    data = env.scene["robot"].data
    return (data.joint_pos - data.default_joint_pos).square().mean(-1)


def joint_pose_rms(env):
    return joint_pose_deviation_cost(env).sqrt()


def joint_velocity_rms(env):
    return env.scene['robot'].data.joint_vel.square().mean(-1).sqrt()


def joint_pose_max_deviation(env):
    data = env.scene["robot"].data
    return (data.joint_pos - data.default_joint_pos).abs().amax(-1)


def head_tilt_cost(env):
    """阈值内按1-cos(倾角)惩罚，阈值外饱和；绕重力轴转动不受罚。"""
    target = head_gravity(env).new_tensor(env.cfg.head_up_axis)
    cosine = -(head_gravity(env) * target).sum(-1)
    return 1 - cosine.clamp(math.cos(env.cfg.head_tilt_limit), 1.)


def head_rotation_cost(env):
    angular = env.scene["robot"].data.body_link_ang_vel_w[:, head_index(env)]
    # 有意偏航按指令跟踪，不以固定朝向或总角速度压制转向。
    error = angular.clone()
    error[:, 2] -= command_term(env).command[:, 2]
    return (error / error.new_tensor((.6, .6, .6))).square().sum(-1).clamp(max=100)


def head_acceleration_cost(env):
    linear = env.scene["robot/head_lin_acc"].data
    angular = env.scene["robot/head_ang_acc"].data
    return ((linear / 15.).square().sum(-1) + (angular / 30.).square().sum(-1)).clamp(max=100)


def ground_force_peak(env):
    force = env.scene["snake_ground_contact"].data.force_history
    return force.norm(dim=-1).flatten(1).amax(-1)


def normalized_ground_peak(env):
    return ground_force_peak(env) / (body_masses(env).sum(-1) * 9.81).clamp_min(1e-6)


def ground_support(env):
    """各几何最强接触之和，仅用于支撑门控，不作为全机合力。"""
    force = env.scene["snake_ground_contact"].data.force
    return force.norm(dim=-1).flatten(1).sum(-1) / (body_masses(env).sum(-1) * 9.81).clamp_min(1e-6)


def impact_cost(env):
    # 正常静载不扣分，关注单个接触的子步峰值；不奖励失去地面支撑。
    return (normalized_ground_peak(env) - 1.5).clamp_min(0).square().clamp(max=100)


def self_collision(env):
    data = env.scene["snake_self_contact"].data
    found = data.found.flatten(1).any(-1)
    transient = data.force_history.norm(dim=-1).flatten(1).amax(-1) > .05
    penetration = data.dist_history.flatten(1).amin(-1) < -1e-4
    return found | transient | penetration


def energy_cost(env):
    data = env.scene["robot"].data
    return (data.actuator_force * data.joint_vel).abs().sum(-1) / 10.


def standstill(env):
    quiet = command_term(env).requested_velocity.norm(dim=-1) < .001
    speed = (planar_velocity(env) / planar_velocity(env).new_tensor((.02, .02, .10))).square().sum(-1)
    joints = env.scene["robot"].data.joint_vel.square().mean(-1)
    return torch.where(quiet, torch.exp(-speed - joints), 0.)


def unsafe(env):
    data = env.scene["robot"].data
    finite = torch.isfinite(data.joint_pos).all(-1) & torch.isfinite(data.joint_vel).all(-1)
    for values in (data.body_link_pos_w, data.body_link_quat_w, data.body_link_lin_vel_w, data.body_link_ang_vel_w):
        finite &= torch.isfinite(values).all(dim=(-1, -2))
    distance = (data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1)
    # 蛇形滚转、抬节换载允许继续探索；仅真正危险状态提前终止。
    return (~finite | (distance > env.cfg.workspace_radius)
            | (normalized_ground_peak(env) > env.cfg.max_contact_bodyweights))


def failure_cost(env):
    return (unsafe(env) | self_collision(env)).float() / env.step_dt


def play_time_out(env):
    return common_mdp.time_out(env) & ~command_term(env).manual_control


def reset_snake(env, env_ids):
    """只在回合重置时写入标准低位蛇形；命令切换不调用此函数。"""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    translation_progress_state(env).reset(ids)
    robot = env.scene["robot"]
    root = robot.data.default_root_state[ids].clone()
    yaw = torch.rand(len(ids), device=env.device) * (2 * torch.pi) if command_term(env).cfg.training else torch.zeros(len(ids), device=env.device)
    yaw_quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
    root[:, :3] = env.scene.env_origins[ids] + root.new_tensor((0., 0., .0555))
    initial_quat = env.cfg.scene.entities["robot"].init_state.rot
    root[:, 3:7] = quat_mul(yaw_quat, root.new_tensor(initial_quat).expand(len(ids), -1))
    root[:, 7:] = 0
    joints = torch.zeros((len(ids), 6), device=env.device)
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
    if env.cfg.motion_frame == 'body_gravity':
        if not hasattr(env, '_snake_forward_w'):
            env._snake_forward_w = torch.zeros((env.num_envs, 2), device=env.device, dtype=root.dtype)
        # foot_0_sole与root同向，避免读取尚未前向更新的site姿态。
        env._snake_forward_w[ids] = _body_motion_forward(root[:, 3:7])[0]


def actor_state(env):
    data = env.scene["robot"].data
    term = command_term(env)
    action = env.action_manager.get_term("joint_pos")
    i = head_index(env)
    gyro = data.body_link_ang_vel_w[:, i]
    if env.cfg.head_gyro_frame == 'head':
        gyro = quat_apply_inverse(data.body_link_quat_w[:, i], gyro)
    elif env.cfg.head_gyro_frame != 'world':
        raise ValueError('未知头部角速度坐标系')
    phase = env.episode_length_buf * env.step_dt * term.cfg.frequency * (2 * torch.pi)
    values = (data.joint_pos, data.joint_vel, action.commanded_target, action.raw_action,
              observation_reference(env), planar_velocity(env), head_gravity(env), gyro,
              head_height(env)[:, None], data.body_link_lin_vel_w[:, i, 2:3],
              torch.stack((phase.sin(), phase.cos()), -1))
    return torch.nan_to_num(torch.cat(values, -1))


def critic_state(env):
    contacts = env.scene["snake_ground_contact"].data.found.flatten(1).float()
    return torch.nan_to_num(torch.cat((actor_state(env), contacts, com_state(env), normalized_ground_peak(env)[:, None]), -1))


def velocity_error(env):
    actual = planar_velocity(env)
    term = command_term(env)
    error = (actual - term.command) / actual.new_tensor(term.cfg.velocity_limits)
    return error.square().mean(-1).sqrt()


def direction_exposure(env, axis, sign):
    cmd = command_term(env).requested_velocity
    others = cmd.clone()
    others[:, axis] = 0
    return ((cmd[:, axis] * sign > .001) & (others.norm(dim=-1) < .001)).float()


def direction_motion_metric(env, axis, sign, error=False):
    """当前纯方向指令组的条件均值；无样本时为零，须结合曝光比例查看。"""
    selected = direction_exposure(env, axis, sign).bool()
    actual = planar_velocity(env)[:, axis]
    values = (actual - command_term(env).command[:, axis]).abs() if error else actual * sign
    mean = torch.where(selected, values, 0.).sum() / selected.sum().clamp_min(1)
    return mean.expand(env.num_envs)


def target_saturation_fraction(env):
    """控制目标距角度边界不足0.001 rad的关节比例，用于惩罚及诊断。"""
    target = env.action_manager.get_term('joint_pos').commanded_target
    limit = env.cfg.actions['joint_pos'].target_limit
    if limit is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (target.abs() >= limit - 1e-3).float().mean(-1)


def action_clip_fraction(env):
    """实际执行动作到达裁剪边界的比例；包含恰好等于边界的动作。"""
    return (env.action_manager.get_term('joint_pos').raw_action.abs() >= 1.).float().mean(-1)


COMMAND_GROUPS = ("translation", "yaw", "stop", "mixed")
GROUP_METRICS = ("fraction", "linear_reward", "yaw_reward", "target_speed",
                 "actual_speed", "linear_error", "yaw_error")


def command_group_mask(env, group):
    """按未平滑请求分组，四组互斥；零指令不混入平移表现。"""
    request = command_term(env).requested_velocity
    linear = request[:, :2].norm(dim=-1) > .001
    yaw = request[:, 2].abs() > .001
    return {"translation": linear & ~yaw, "yaw": ~linear & yaw,
            "stop": ~linear & ~yaw, "mixed": linear & yaw}[group]


def command_group_metric(env, group, metric):
    """每环境掩码分子；标准日志补充相同口径的fraction比值作为条件均值。

    保留每环境样本后交给MetricsManager累计，避免广播瞬时组均值造成
    异步复位偏差。奖励含配置权重，不乘dt；误差跟随训练的平滑指令。
    """
    selected = command_group_mask(env, group)
    if metric == "fraction":
        return selected.float()
    if metric in ("linear_reward", "yaw_reward"):
        name = "translation_tracking" if metric == "linear_reward" else "yaw_tracking"
        cfg = env.cfg.rewards[name]
        values = cfg.func(env, **cfg.params) * cfg.weight
    else:
        actual = planar_velocity(env)
        target = command_term(env).command
        if metric == "target_speed":
            values = target[:, :2].norm(dim=-1)
        elif metric == "actual_speed":
            values = actual[:, :2].norm(dim=-1)
        elif metric == "linear_error":
            values = (actual[:, :2] - target[:, :2]).norm(dim=-1)
        elif metric == "yaw_error":
            values = (actual[:, 2] - target[:, 2]).abs()
        else:
            raise ValueError(f"未知指令组指标：{metric}")
    return torch.where(selected, values, 0.)


def translation_curriculum_metric(env, metric):
    term = command_term(env)
    value = (getattr(term, "current_translation_cap", term.cfg.velocity_limits[0])
             if metric == "cap" else getattr(term, "translation_stage", 0))
    return term.command.new_full((env.num_envs,), float(value))
