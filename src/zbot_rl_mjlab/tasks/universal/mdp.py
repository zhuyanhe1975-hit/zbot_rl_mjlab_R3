"""综合任务状态、模式门控奖励与允许低位接地的终止条件。"""
from typing import cast

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.utils.lab_api.math import quat_apply

from ..common import constant
from ..transformation import mdp as transform
from ..transformation import snake_to_biped as standing
from ..transformation.poses import SNAKE_JOINT_POS, SNAKE_ROOT_POS, SNAKE_ROOT_QUAT
from ..transformation.roll_reset import collision_radial_envelope, rolled_root_states
from .commands import UniversalCommand, UniversalCommandCfg

UNIVERSAL_ACTOR_DIM = 50
UNIVERSAL_CRITIC_DIM = 53


def command_term(env) -> UniversalCommand:
    return cast(UniversalCommand, env.command_manager.get_term("transform"))


def base_height(env):
    data, i = transform._base(env)
    return data.body_link_pos_w[:, i, 2] - env.scene.env_origins[:, 2]


def heading(env):
    """腿走使用基座前向；首版蛇教师实验使用归档世界+X指令参考。"""
    if getattr(env.cfg, 'omni_site_frame', False):
        from .omni import site_heading

        return site_heading(env)
    data, i = transform._base(env)
    axis = torch.zeros_like(data.body_link_pos_w[:, i])
    axis[:, 2] = 1
    base_z = quat_apply(data.body_link_quat_w[:, i], axis)
    projected = torch.stack((base_z[:, 1], -base_z[:, 0]), -1)
    norm = projected.norm(dim=-1, keepdim=True)
    valid = torch.isfinite(projected).all(-1) & (norm[:, 0] > .15)
    term = command_term(env)
    if getattr(env.cfg, 'snake_teacher_enabled', False):
        # 教师优化的是世界+X速度；头部摆动不能把前爬请求旋转成世界侧移。
        snake_forward = torch.zeros_like(projected)
        snake_forward[:, 0] = 1
        projected = torch.where(term.standing_mode[:, None], projected, snake_forward)
        norm = projected.norm(dim=-1, keepdim=True)
        valid = torch.isfinite(projected).all(-1) & (norm[:, 0] > .15)
    term.heading.copy_(torch.where(valid[:, None], projected / norm.clamp_min(1e-6), term.heading))
    first = valid & ~term.heading_initialized
    # torch.where 避免布尔索引赋值在每步触发设备同步。
    term.initial_heading.copy_(torch.where(first[:, None], term.heading, term.initial_heading))
    term.heading_initialized |= valid
    return term.heading


def velocity(env):
    if getattr(env.cfg, 'omni_site_frame', False):
        from .omni import site_velocity

        return site_velocity(env)
    data, i = transform._base(env)
    forward = heading(env)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    linear = data.body_link_lin_vel_w[:, i, :2]
    return torch.stack(((linear * forward).sum(-1), (linear * lateral).sum(-1), data.body_link_ang_vel_w[:, i, 2]), -1)


def tracking_velocity(env):
    """独立 site 蛇任务跟踪 foot_0_sole；其余任务保留原来的质心参考。"""
    if getattr(env.cfg, 'site_snake_omni', False):
        from .omni import site_velocity

        return site_velocity(env)
    if getattr(env.cfg, 'omni_site_frame', False):
        from .omni import tracking_velocity as site_tracking_velocity

        return site_tracking_velocity(env)
    data, i = transform._base(env)
    masses = transform.body_masses(env)
    linear = (data.body_com_lin_vel_w[..., :2] * masses[..., None]).sum(-2)
    linear /= masses.sum(-1, keepdim=True).clamp_min(1e-6)
    forward = heading(env)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    return torch.stack(((linear * forward).sum(-1), (linear * lateral).sum(-1), data.body_link_ang_vel_w[:, i, 2]), -1)


def contacts(env):
    found = env.scene["feet_ground_contact"].data.found
    feet = found.reshape(env.num_envs, 2, -1).any(-1).float()
    base = env.scene["base_ground_contact"].data.found.reshape(env.num_envs, -1).any(-1).float()
    return torch.cat((feet, base[:, None]), -1)


def configuration_ready(env, relaxed=False):
    """构型而非静止关节判据；允许摆腿和蛇形行波。"""
    term = command_term(env)
    h = base_height(env)
    upright = standing.base_gravity_alignment_cost(env)
    supported = contacts(env)[:, :2].any(-1)
    startup = torch.zeros_like(supported)
    reset_step = getattr(term, 'canonical_motion_start', None)
    if relaxed and reset_step is not None:
        # Canonical reset poses begin a few millimetres above ground. Contact
        # sensors need physics steps to populate; this is not a failed transform.
        age = (env.common_step_counter - reset_step) * env.step_dt
        startup = (reset_step >= 0) & (age < getattr(term.cfg, 'motion_startup_s', 0.))
        supported |= startup
    stand = (h > (.14 if relaxed else .20)) & (upright < (.9 if relaxed else .45)) & supported
    height = getattr(term.cfg, 'standing_ready_height', None)
    tolerance = getattr(term.cfg, 'transition_joint_tolerance', None)
    if not relaxed and height is not None:
        stand &= h >= height
    if not relaxed and tolerance is not None:
        goal = constant(env.scene['robot'].data.joint_pos, env.cfg.standing_joint_pos)
        close = (env.scene['robot'].data.joint_pos - goal).abs().amax(-1) <= tolerance
        # Only a transformation must settle near its target; walking remains free to step.
        stand &= ~term.transition_mask | close
    snake = (h < (.18 if relaxed else .12)) & (transform.com_state(env)[:, 0] < (.16 if relaxed else .10))
    snake &= (normalized_force(env) > .1) | startup
    ready = torch.where(term.standing_mode, stand, snake)
    straightening = getattr(term, 'straightening_mask', None)
    return ready if straightening is None else ready & ~straightening


def gait_observation(env):
    """教师和学生共享确定性的步态相位与频率。"""
    term = command_term(env)
    frequency = term.gait_frequency
    if getattr(env.cfg, 'snake_teacher_enabled', False):
        # 蛇归档时钟为0.7Hz；不能向教师隐藏学生不可见的另一套相位。
        frequency = torch.where(term.standing_mode, frequency, .7)
    phase = env.episode_length_buf * env.step_dt * frequency * (2 * torch.pi)
    return torch.stack((phase.sin(), phase.cos(), frequency), -1)


def heading_error(env):
    """以初始有效前向为零点，符号沿用既有行走教师。"""
    forward = heading(env)
    initial = command_term(env).initial_heading
    cross = forward[:, 0] * initial[:, 1] - forward[:, 1] * initial[:, 0]
    return torch.atan2(cross, (forward * initial).sum(-1))


def actor_state(env):
    data, i = transform._base(env)
    term = command_term(env)
    action = env.action_manager.get_term("joint_pos")
    values = (data.joint_pos, data.joint_vel, action.commanded_target, action.raw_action,
              velocity(env), data.body_link_lin_vel_w[:, i, 2:3], transform.projected_gravity(env), data.body_link_ang_vel_w[:, i],
              base_height(env)[:, None], contacts(env), getattr(term, 'observation_mode', term.command[:, 0])[:, None],
              torch.nn.functional.one_hot(term.phase, 4).float(), term.velocity_command,
              gait_observation(env), heading_error(env)[:, None])
    return torch.nan_to_num(torch.cat(values, -1))


def force_peak(env):
    force = env.scene["transform_ground_contact"].data.force_history
    return force.norm(dim=-1).flatten(1).amax(-1)


def normalized_force(env):
    mass = transform.body_masses(env).sum(-1)
    gravity = sum(v * v for v in env.cfg.sim.mujoco.gravity) ** .5
    return force_peak(env) / (mass * gravity).clamp_min(1e-6)


def critic_state(env):
    return torch.nan_to_num(torch.cat((actor_state(env), transform.com_state(env), normalized_force(env)[:, None]), -1))


def motion_tracking_score(actual, requested, limits=(.25, .20, .8)):
    """按相对指令幅值汇总各轴推进，未指令轴仅收取独立小幅漂移成本。"""
    limits = constant(requested, limits)
    scale = torch.maximum(requested.abs(), limits * .1)
    kernel = 1 / (1 + 4 * ((actual - requested) / scale).square())
    stationary = 1 / (1 + 4 * (requested / scale).square())
    progress = (kernel - stationary) / (1 - stationary).clamp_min(.2)
    active = requested.abs() > 1e-6
    weights = torch.where(active, requested.abs() / limits, 0.)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    advance = (progress * weights).sum(-1)
    # 零指令轴的正常侧摆不应淹没有效推进；仍惩罚只有摆动没有前进。
    drift = (actual / (limits * .4)).square().clamp(max=1000)
    drift = torch.where(active, 0., drift).sum(-1) / (~active).sum(-1).clamp_min(1)
    return torch.where(requested.norm(dim=-1) > .01, advance - .1 * drift, 0.)


def velocity_tracking(env):
    term = command_term(env)
    return torch.where(term.transition_mask, 0., motion_tracking_score(tracking_velocity(env), term.velocity_command, term.cfg.velocity_limits))


def posture_reward(env):
    term = command_term(env)
    h = base_height(env)
    upright = standing.base_gravity_alignment_cost(env)
    stand = torch.exp(-((h - env.cfg.standing_base_height) / .08).square() - 2 * upright)
    low = torch.exp(-((transform.com_state(env)[:, 0] - .055) / .075).square())
    low *= (normalized_force(env) > .1).float()
    # 蛇形不约束零关节，不强迫所有刚体贴地。
    return torch.where(term.standing_mode, stand, low)


def transition_progress_reward(env):
    """姿态势能的有符号进步；到位后继续计算，避免阶段切换奖励断崖。"""
    term = command_term(env)
    potential = posture_reward(env)
    fresh = term.potential_step != env.common_step_counter
    progress = torch.where(torch.isfinite(term.potential), (potential - term.potential) / env.step_dt, 0.)
    term.progress_reward.copy_(torch.where(fresh, progress, term.progress_reward))
    term.potential.copy_(torch.where(fresh, potential, term.potential))
    term.potential_step.masked_fill_(fresh, env.common_step_counter)
    return term.progress_reward


def standstill_reward(env):
    term = command_term(env)
    quiet = term.requested_velocity.norm(dim=-1) < .01
    speed = velocity(env).square().sum(-1)
    joints = env.scene["robot"].data.joint_vel.square().mean(-1)
    return torch.where(quiet & ~term.transition_mask, torch.exp(-speed / .0025 - joints / .25), 0.)


def energy_cost(env):
    data = env.scene["robot"].data
    return torch.nan_to_num((data.actuator_force * data.joint_vel).abs().sum(-1)).clamp(max=1000)


def action_rate_cost(env):
    manager = env.action_manager
    return (manager.action - manager.prev_action).square().sum(-1)


def impact_cost(env):
    return (normalized_force(env) - 2.).clamp_min(0).square().clamp(max=1000)


def unsafe(env):
    data, i = transform._base(env)
    finite = torch.isfinite(data.joint_pos).all(-1) & torch.isfinite(data.joint_vel).all(-1)
    finite &= torch.isfinite(data.body_link_pos_w).all(dim=(-1, -2))
    finite &= torch.isfinite(data.body_link_lin_vel_w).all(dim=(-1, -2))
    finite &= torch.isfinite(data.body_link_ang_vel_w).all(dim=(-1, -2))
    finite &= torch.isfinite(data.body_link_quat_w).all(dim=(-1, -2))
    distance = (data.body_link_pos_w[:, i, :2] - env.scene.env_origins[:, :2]).norm(dim=-1)
    term = command_term(env)
    # 仅已进入双足运动阶段时检测真正塌落；下降和蛇形低位合法。
    fallen = term.standing_mode & term.standing_achieved & (base_height(env) < .10)
    timed_out = term.transition_mask & ((env.common_step_counter - term.start_step) * env.step_dt >= term.cfg.transition_timeout_s)
    automatic_failure = (distance > env.cfg.workspace_radius) | fallen | timed_out
    if not term.cfg.training:
        automatic_failure &= ~term.manual_control
    return ~finite | automatic_failure


def failure_cost(env):
    """消除奖励管理器的 dt 缩放，使每次失败固定扣分。"""
    return unsafe(env).float() / env.step_dt


def demonstration_time_out(env):
    """运动演示最长十秒，变形到位并短暂保持后及时重新演示。"""
    term = command_term(env)
    elapsed = env.episode_length_buf * env.step_dt
    groups = torch.arange(env.num_envs, device=env.device) % (5 if term.cfg.snake_teacher_enabled else 4)
    transformation = (groups == 2) | (groups == 3)
    completed = transformation & ~term.transition_mask & (elapsed >= 1.5)
    return ((elapsed >= 10.) | completed) & term.cfg.training & term.pretraining & ~term.manual_control


def play_time_out(env):
    """手动控制保留连续轨迹，数值失效仍由 unsafe 独立处理。"""
    return common_mdp.time_out(env) & ~command_term(env).manual_control


def reset_universal(env, env_ids):
    """只在环境重置时混合双足与蛇形，手动切换不调用此函数。"""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene["robot"]
    training = cast(UniversalCommandCfg, env.cfg.commands["transform"]).training
    term = command_term(env)
    choose = torch.rand(len(ids), device=env.device) < .5 if training else torch.ones(len(ids), device=env.device, dtype=torch.bool)
    if training and term.pretraining:
        groups = ids % (5 if term.cfg.snake_teacher_enabled else 4)
        choose = (groups != 3) & (groups != 4)
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] = env.scene.env_origins[ids] + root.new_tensor(SNAKE_ROOT_POS)
    root[:, 3:7] = root.new_tensor(SNAKE_ROOT_QUAT)
    if training and term.pretraining:
        # 起立预热覆盖评测范围内的侧滚，仍保持教师适用的镜像关节初态。
        radial, low, high = collision_radial_envelope(tuple(SNAKE_JOINT_POS))
        roll = (torch.rand(len(ids), device=env.device) * 2 - 1) * (40 * torch.pi / 180)
        relative = root.clone()
        relative[:, :3] -= env.scene.env_origins[ids]
        root = rolled_root_states(relative, env.scene.env_origins[ids], roll,
                                  root.new_tensor(radial), low, high, .003)
    root[:, 7:] = 0
    q = root.new_tensor(SNAKE_JOINT_POS).repeat(len(ids), 1)
    q[choose] = q.new_tensor(env.cfg.standing_joint_pos)
    root[choose, :3] = env.scene.env_origins[ids[choose]] + root.new_tensor([0., -.06, .003 - collision_radial_envelope(tuple(env.cfg.standing_joint_pos))[1]])
    root[choose, 3:7] = root.new_tensor([1., 0., 0., 0.])
    if training and getattr(env.cfg, 'snake_teacher_enabled', False):
        # 留出独立的规范前爬初态，不改变起立教师的滚转姿态库。
        crawl = (ids % 5 == 4) if term.pretraining else (~choose & ~term.standing_mode[ids])
        root[crawl, :3] = env.scene.env_origins[ids[crawl]] + root.new_tensor([0., 0., .0555])
        root[crawl, 3:7] = root.new_tensor([2 ** -.5, 0., -(2 ** -.5), 0.])
        q[crawl] = 0
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    robot.write_root_state_to_sim(root, env_ids=ids)


def phase_fraction(env, phase: int):
    """每环境阶段标记；回合结束子集的日志均值不代表全局采样占比。"""
    return (command_term(env).phase == phase).float()


def mode_motion_error(env, phase: int):
    """有效运动样本的归一化速度误差；空样本返回有限零值。

    按有效样本比例补偿，使管理器的批均值等于所选模式内均值。
    需结合相应 phase_fraction 判断该批次是否覆盖此模式。
    """
    term = command_term(env)
    selected = (term.phase == phase) & (term.requested_velocity.norm(dim=-1) > .01)
    limits = constant(term.velocity_command, term.cfg.velocity_limits)
    error = ((tracking_velocity(env) - term.velocity_command) / limits).square().mean(-1).sqrt()
    scale = selected.float().mean().clamp_min(1. / env.num_envs)
    return torch.where(selected, error / scale, 0.)
