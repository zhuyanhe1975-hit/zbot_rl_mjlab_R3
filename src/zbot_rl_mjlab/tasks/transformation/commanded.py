"""由站立（+1）／蛇形（-1）指令控制的单个镜像教师策略。"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ..common import constant
from . import mdp, robust
from . import snake_to_biped as standing
from .roll_reset import collision_radial_envelope
from .teacher_observations import teacher_actor_state, teacher_critic_state

TASK_ID = "Mjlab-Zbot-6dof-Transform-Snake-Biped-Bidirectional"


@dataclass
class BidirectionalEnvCfg(robust.RobustSnakeToBipedEnvCfg):
    initial_standing_fraction: float = 0.5
    standing_pose_gravity_scale: float = 1.0
    initial_roll_range_deg: tuple[float, float] = (-180.0, 180.0)
    fall_retry_arm_height: float = 0.05
    fall_retry_drop_threshold: float = 0.01
    early_contact_arm_height: float = 0.05
    early_contact_drop_threshold: float = 0.01


@dataclass(kw_only=True)
class TransformCommandCfg(CommandTermCfg):
    # 从采样起始姿态的相反目标开始；成功后自动交替目标。
    mode: str = "source"
    auto_switch_on_success: bool = True
    auto_switch_joint_tolerance: float = 0.15

    def build(self, env):
        return TransformCommand(self, env)


class TransformCommand(CommandTerm):
    _env: Any  # 环境保存本任务按需分配的重置状态。
    cfg: TransformCommandCfg

    def __init__(self, cfg: TransformCommandCfg, env):
        super().__init__(cfg, env)
        if cfg.mode not in ("source", "stand", "snake"):
            raise ValueError("transform command mode must be source, stand or snake")
        if not math.isfinite(cfg.auto_switch_joint_tolerance) or cfg.auto_switch_joint_tolerance <= 0:
            raise ValueError("auto_switch_joint_tolerance must be finite and positive")
        lo, hi = cfg.resampling_time_range
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
            raise ValueError("command resampling times must be finite and positive")
        self._command = torch.ones(self.num_envs, 1, device=self.device)
        self.start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.manual_control = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.switch_hold_time = torch.zeros(self.num_envs, device=self.device)
        self.metrics["completed_transitions"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        self.manual_control[env_ids] = False
        if self.cfg.mode == "source":
            starts_standing = self._env._transform_initial_standing[env_ids]
            self._command[env_ids, 0] = torch.where(starts_standing, -1.0, 1.0)
        else:
            self._command[env_ids, 0] = 1.0 if self.cfg.mode == "stand" else -1.0
        self.start_step[env_ids] = self._env.common_step_counter
        clear_command_memory(self._env, env_ids)

    def _update_metrics(self):
        pass

    def _update_command(self, env_ids):
        # MJLab 在奖励／指标计算之后、新观测生成之前调用。
        # 仅针对重置环境的更新绝不能推进其他环境。
        if env_ids is not None or not self.cfg.auto_switch_on_success or self.cfg.mode != "source":
            return
        ready = (self.switch_hold_time >= self._env.cfg.success_hold_s - 1e-5)
        ready &= auto_switch_condition(self._env) & ~self.manual_control
        reset_buf = getattr(self._env, "reset_buf", None)
        if reset_buf is not None:
            ready &= ~reset_buf
        ids = ready.nonzero().flatten()
        if not len(ids):
            return
        self._command[ids] *= -1
        self.start_step[ids] = self._env.common_step_counter
        self.time_left[ids] = math.inf
        self.command_counter[ids] += 1
        self.metrics["completed_transitions"][ids] += 1
        clear_command_memory(self._env, ids)

    def set_target(self, target: str, env_ids=None):
        """原地切换目标，不写入机器人姿态或动作目标。"""
        if target not in ("stand", "snake"):
            raise ValueError("target must be stand or snake")
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        value = 1.0 if target == "stand" else -1.0
        changed = ids[self._command[ids, 0] != value]
        self._command[ids, 0] = value
        self.time_left[ids] = math.inf
        self.manual_control[ids] = True
        self.start_step[changed] = self._env.common_step_counter
        clear_command_memory(self._env, changed)



def play_time_out(env):
    """键盘指令开始手动控制之前，保持自动演示。"""
    term = env.command_manager.get_term("transform")
    return common_mdp.time_out(env) & ~term.manual_control



def play_failure(env, condition):
    """键盘指令即使在倒下后也保留连续的物理轨迹。"""
    triggered = condition(env)
    term = env.command_manager.get_term("transform")
    return triggered & ~term.manual_control


def command(env):
    return env.command_manager.get_command("transform")


def wants_stand(env):
    return command(env)[:, 0] > 0


def goal_joints(env):
    positions = env.scene["robot"].data.joint_pos
    return wants_stand(env)[:, None] * constant(positions, env.cfg.standing_joint_pos)


def clear_command_memory(env, ids):
    """清除目标相关历史，同时保留实际动力学状态。"""
    if not len(ids):
        return
    task = robust.fall_retry_state(env)
    transition = mdp.state(env)
    with torch.inference_mode():
        task.reset(ids, env.common_step_counter)
        transition.reset(ids, env.common_step_counter)
        env.command_manager.get_term("transform").switch_hold_time[ids] = 0
        if hasattr(env, "_standing_goal_valid"):
            env._standing_goal_valid[ids] = False
        # 交互回放时，指令可能在两个物理帧之间改变。
        env._observation_generation = getattr(env, "_observation_generation", 0) + 1



def sample_initial_standing(env, ids):
    """根据当前实际目标方向分配重置名额。

    失败的短站立回合不应变成长下降回合，进而逐步挤掉采样缓冲区中的站立样本。
    通过随机排序，选择需要重置的环境来填补当前空缺的方向名额。
    """
    fraction = env.cfg.initial_standing_fraction
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("initial_standing_fraction must be in [0, 1]")
    mode = env.cfg.commands["transform"].mode
    if mode != "source":
        return torch.full((len(ids),), mode == "snake", dtype=torch.bool, device=env.device)
    if env.num_envs == 1:
        return torch.rand(len(ids), device=env.device) < fraction
    # 起始标志描述重置姿态，成功翻转后便会过时。
    # 站立起始姿态对应蛇形目标，因此统计当前蛇形目标数。
    current = command(env)[:, 0] < 0
    remaining = current.sum() - current[ids].sum()
    desired = int(env.num_envs * fraction + 0.5)
    vacancies = (desired - remaining).clamp(0, len(ids))
    return torch.randperm(len(ids), device=env.device) < vacancies


def reset_command_transition(env, env_ids):
    """混合起始姿态；随后由指令管理器采样相反目标。"""
    fraction = env.cfg.initial_standing_fraction
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("initial_standing_fraction must be in [0, 1]")
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    # 先在调用方上下文中分配持久缓冲区，再允许写入目标掩码；
    # 这些掩码可能由仅推理的策略步生成。
    robust.fall_retry_state(env)
    mdp.state(env)
    with torch.inference_mode():
        robust.reset_fall_retry(env, ids)
    if not hasattr(env, "_transform_initial_standing"):
        env._transform_initial_standing = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    choose = sample_initial_standing(env, ids)
    env._transform_initial_standing[ids] = choose
    standing_ids = ids[choose]
    if not len(standing_ids):
        return
    robot = env.scene["robot"]
    root = robot.data.default_root_state[standing_ids].clone()
    # 根部为 foot_0：单位四元数对应归档中的直立站立方向。
    root[:, :3] = env.scene.env_origins[standing_ids] + root.new_tensor([0, -0.06, 0])
    if not hasattr(env, "_standing_reset_min_z"):
        env._standing_reset_min_z = collision_radial_envelope(tuple(env.cfg.standing_joint_pos))[1]
    root[:, 2] += env.cfg.initial_roll_clearance - env._standing_reset_min_z
    root[:, 3:7] = root.new_tensor([1, 0, 0, 0])
    root[:, 7:] = 0
    joints = root.new_tensor(env.cfg.standing_joint_pos).expand(len(standing_ids), -1).clone()
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=standing_ids)
    robot.write_root_state_to_sim(root, env_ids=standing_ids)
    with torch.inference_mode():
        env._initial_snake_roll_offset[standing_ids] = 0


def actor_state(env):
    # 原有 36 列保持不变，包括固定站立参考。
    return torch.cat((teacher_actor_state(env), command(env)), dim=-1)


def critic_state(env):
    # 保留公共教师状态顺序；末尾追加显式方向指令。
    return torch.cat((teacher_critic_state(env), command(env)), dim=-1)


def pose_reward(env):
    scale = env.cfg.joint_standing_error_scale
    error = (env.scene["robot"].data.joint_pos - goal_joints(env)).abs()
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("joint_standing_error_scale must be finite and positive")
    gravity_scale = env.cfg.standing_pose_gravity_scale
    if not math.isfinite(gravity_scale) or gravity_scale < 0:
        raise ValueError("standing_pose_gravity_scale must be finite and nonnegative")
    cost = torch.where(error < scale, 0.5 * error.square() / scale, error - 0.5 * scale)
    joint_cost = torch.nan_to_num(cost, nan=100., posinf=100.).mean(-1)
    # 仅靠关节角无法区分直立双足与躺倒姿态。
    gravity_cost = standing.base_gravity_alignment_cost(env)
    return -(joint_cost + torch.where(wants_stand(env), gravity_scale * gravity_cost, 0.0))


def height_reward(env):
    progress = standing.base_height_progress_reward(env)
    # 站立采用完整高度；蛇形采用既有低位蛇形高度。
    return torch.where(wants_stand(env), progress, (1 - progress).clamp(-1, 1))


def velocity_reward(env, target_speed=0.15, pose_error_range=(0.2, 0.8)):
    # 保留一次性高度锁存，但不能让折叠躺倒姿态
    # 仅因匹配目标关节角就消除上升驱动力。
    value = robust.initial_ascent_velocity_reward(env, target_speed, None)
    if pose_error_range is not None:
        zero_error, full_error = pose_error_range
        if not (math.isfinite(zero_error) and math.isfinite(full_error) and 0 <= zero_error < full_error):
            raise ValueError("pose_error_range must satisfy 0 <= zero < full, both finite")
        error = (env.scene["robot"].data.joint_pos - goal_joints(env)).abs().amax(-1)
        fraction = ((error - zero_error) / (full_error - zero_error)).clamp(0, 1)
        pose_ready = 1 - fraction.square() * (3 - 2 * fraction)
        span = env.cfg.standing_base_height - standing.SNAKE_ROOT_POS[2]
        if span <= 0:
            raise ValueError("standing_base_height must exceed snake height")
        lift_ready = ((standing.base_height(env) - standing.SNAKE_ROOT_POS[2]) / span).clamp(0, 1)
        upright_ready = (1 - 0.5 * standing.base_gravity_alignment_cost(env)).clamp(0, 1)
        readiness = torch.nan_to_num(pose_ready * lift_ready * upright_ready, nan=0.)
        value = value * (1 - readiness)
    return torch.where(wants_stand(env), value, 0.0)


def sole_reward(env, angle_power=1.0):
    return torch.where(wants_stand(env), standing.sole_ground_alignment_reward(env, angle_power), 0.0)


def stand_finish_joint_speed_cost(env):
    """仅当站立基座足够高时，才惩罚实际关节速度。"""
    finishing = wants_stand(env) & (standing.base_height(env) >= env.cfg.ascent_reward_stop_height)
    cost = env.scene["robot"].data.joint_vel.square().mean(-1)
    return torch.where(finishing, cost, 0.0)


def ground_force_peak(env):
    """策略步全部子步中各刚体地面合力的峰值。"""
    history = env.scene["transform_ground_contact"].data.force_history
    return history.norm(dim=-1).flatten(1).amax(-1)


def descent_ground_impact_cost(env, max_body_weights=2.0):
    """在蛇形指令期间惩罚超额地面作用力，正常支撑不受惩罚。"""
    if not math.isfinite(max_body_weights) or max_body_weights <= 1:
        raise ValueError("max_body_weights must be finite and greater than one")
    gravity = math.sqrt(sum(component**2 for component in env.cfg.sim.mujoco.gravity))
    if not math.isfinite(gravity) or gravity <= 0:
        raise ValueError("ground impact normalization requires nonzero finite gravity")
    weight = mdp.body_masses(env).sum(-1) * gravity
    cost = (ground_force_peak(env) / weight - max_body_weights).clamp_min(0).square()
    return torch.where(wants_stand(env), 0.0, cost)



def final_condition(env, joint_tolerance=None):
    return torch.where(wants_stand(env), standing.final_condition(env, joint_tolerance), mdp.final_condition(env, joint_tolerance))


def auto_switch_condition(env):
    tolerance = env.command_manager.get_term("transform").cfg.auto_switch_joint_tolerance
    return final_condition(env, joint_tolerance=tolerance)


def update_state(env):
    task = mdp.state(env)
    fresh = task.last_step != env.common_step_counter
    valid = final_condition(env)
    task.advance_hold(valid, env.common_step_counter, env.step_dt, commit=False)
    term = env.command_manager.get_term("transform")
    switch_valid = auto_switch_condition(env) & ~term.manual_control
    switch_held = torch.where(switch_valid, term.switch_hold_time + env.step_dt, 0.0)
    term.switch_hold_time.copy_(torch.where(fresh, switch_held, term.switch_hold_time))
    task.last_step.fill_(env.common_step_counter)
    env._standing_goal_valid = valid
    env._standing_goal_step = env.common_step_counter
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def success(env):
    return (mdp.state(env).hold_time >= env.cfg.success_hold_s - 1e-5) & final_condition(env)


def fallen(env):
    """低高度时启用检测，但重置前要求相对峰值发生实际下降。"""
    cfg = env.cfg
    for name in ("fall_retry_arm_height", "fall_retry_height", "fall_retry_drop_threshold", "fall_retry_hold_s"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg.fall_retry_arm_height >= cfg.standing_base_height:
        raise ValueError("fall_retry_arm_height must be below standing_base_height")
    state = robust.fall_retry_state(env)
    fresh = state.last_step != env.common_step_counter
    height = standing.base_height(env)
    peak = torch.maximum(state.peak_base_height, height)
    armed = state.armed | (wants_stand(env) & ~state.recovering & (height >= cfg.fall_retry_arm_height))
    dropped = peak - height >= cfg.fall_retry_drop_threshold
    low = armed & dropped & (height < cfg.fall_retry_height)
    duration = torch.where(low, state.low_time + env.step_dt, 0.0)
    state.armed.copy_(torch.where(fresh, armed, state.armed))
    state.peak_base_height.copy_(torch.where(fresh, peak, state.peak_base_height))
    state.low_time.copy_(torch.where(fresh, duration, state.low_time))
    state.last_step.fill_(env.common_step_counter)
    return wants_stand(env) & ~state.recovering & state.armed & (state.low_time >= cfg.fall_retry_hold_s - 1e-6)


def early_contact(env):
    """必须先出现显著上升和下降，再判定基座接触失败。"""
    cfg = env.cfg
    if not (math.isfinite(cfg.early_contact_arm_height) and 0 < cfg.early_contact_arm_height < cfg.standing_base_height):
        raise ValueError("early_contact_arm_height must be positive and below standing_base_height")
    for name in ("early_contact_motion_threshold", "early_contact_drop_threshold", "early_contact_min_down_speed"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    state = robust.fall_retry_state(env)
    fresh = state.contact_last_step != env.common_step_counter
    contact = robust.base_ground_contact(env)
    height = standing.base_height(env)
    motion = state.motion_started | (mdp.commanded_target(env).abs().amax(-1) >= cfg.early_contact_motion_threshold)
    separated = state.base_left_ground | (motion & state.had_base_contact & ~contact)
    peak = torch.maximum(state.peak_base_height, height)
    meaningful_fall = (peak >= cfg.early_contact_arm_height) & (peak - height >= cfg.early_contact_drop_threshold)
    recontact_or_impact = separated | (standing.base_vertical_velocity(env) <= -cfg.early_contact_min_down_speed)
    detected = wants_stand(env) & motion & contact & meaningful_fall & recontact_or_impact & ~state.recovering
    fallen = state.early_base_fallen | detected
    for current, updated in (
        (state.motion_started, motion), (state.had_base_contact, state.had_base_contact | contact),
        (state.base_left_ground, separated), (state.peak_base_height, peak),
        (state.early_base_fallen, fallen),
    ):
        current.copy_(torch.where(fresh, updated, current))
    state.contact_last_step.fill_(env.common_step_counter)
    return state.early_base_fallen & wants_stand(env)


def abandoned_exploration(env):
    task = robust.fall_retry_state(env)
    data, base_id = mdp._base(env)
    cfg = env.cfg
    desired_height = torch.where(wants_stand(env), cfg.standing_base_height, standing.SNAKE_ROOT_POS[2])
    near_goal = (standing.base_height(env) - desired_height).abs() < cfg.stagnation_height_tolerance
    near_goal &= (data.joint_pos - goal_joints(env)).abs().amax(-1) < cfg.stagnation_joint_tolerance
    quiet = data.joint_vel.square().mean(-1).sqrt() < cfg.stagnation_joint_speed
    quiet &= data.body_link_lin_vel_w[:, base_id].norm(dim=-1) < cfg.stagnation_base_speed
    quiet &= data.body_link_ang_vel_w[:, base_id].norm(dim=-1) < cfg.stagnation_angular_speed
    term = env.command_manager.get_term("transform")
    elapsed = (env.common_step_counter - term.start_step) * env.step_dt
    # 使用指令持续时间而非回合持续时间，以便切换后允许重新进入动作。
    eligible = elapsed >= cfg.stagnation_grace_s
    fresh = task.still_last_step != env.common_step_counter
    duration = torch.where(quiet & ~near_goal & eligible, task.still_time + env.step_dt, 0.0)
    task.still_time.copy_(torch.where(fresh, duration, task.still_time))
    task.still_last_step.fill_(env.common_step_counter)
    return task.still_time > cfg.stagnation_hold_s + 1e-6


def joint_error(env):
    return (env.scene["robot"].data.joint_pos - goal_joints(env)).abs().mean(-1)


def command_metric(env):
    return command(env)[:, 0]


def stand_env_fraction(env):
    # 将全局占比复制到各环境，避免按回合记录日志时
    # 因回合长度悬殊而产生偏差。
    return wants_stand(env).float().mean().expand(env.num_envs)
