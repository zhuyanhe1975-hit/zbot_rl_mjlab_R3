"""与检查点兼容的蛇形转双足力／力矩扰动微调。"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from mjlab.envs import mdp as common_mdp

from ..common import PartialBaseImpulse
from . import mdp as transformation_mdp
from .actions import (
    MirroredJointTargetRateAction,
    MirroredJointTargetRateActionCfg,
)
from .mdp import commanded_target
from .mdp import final_condition as snake_final_condition
from .roll_reset import randomize_initial_snake_roll
from .snake_to_biped import (
    BETA_TASK_ID,
    SnakeToBipedEnvCfg,
    base_ascent_velocity_reward,
    base_height,
    base_vertical_velocity,
    reset_reverse_transition,
)
from .snake_to_biped import (
    final_condition as standing_final_condition,
)
from .snake_to_biped import (
    update_state as update_standing_state,
)

ROBUST_TASK_ID = BETA_TASK_ID + "-Robust"
PRETRAINED_CHECKPOINT = str(
    Path(__file__).resolve().parents[4]
    / "bests/transform_snake_to_biped_beta_2026-10-17_iter2550/model_2550.pt"
)


@dataclass
class RobustSnakeToBipedEnvCfg(SnakeToBipedEnvCfg):
    compact_standing_rewards: ClassVar[bool] = True
    pretrained_checkpoint: str | None = None
    disturbance_initial_scale: float = 1.0
    disturbance_warmup_steps: int = 0
    disturbance_ramp_steps: int = 0
    ascent_reward_stop_height: float = 0.23
    fall_retry_arm_height: float = 0.18
    fall_retry_height: float = 0.10
    fall_retry_hold_s: float = 0.10
    recovery_target_speed: float = 0.5
    recovery_snake_hold_s: float = 0.5
    recovery_target_tolerance: float = 0.02
    initial_roll_range_deg: tuple[float, float] = (-90.0, 90.0)
    initial_roll_clearance: float = 0.0001
    early_contact_motion_threshold: float = 0.01
    early_contact_drop_threshold: float = 0.001
    early_contact_min_down_speed: float = 0.03
    stagnation_height_tolerance: float = 0.03
    stagnation_joint_tolerance: float = 0.25
    stagnation_joint_speed: float = 0.03
    stagnation_base_speed: float = 0.01
    stagnation_angular_speed: float = 0.05
    stagnation_hold_s: float = 1.0
    stagnation_grace_s: float = 0.3


class FallRetryState:
    def __init__(self, env):
        self.armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.low_time = torch.zeros(env.num_envs, device=env.device)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.recovering = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.snake_time = torch.zeros(env.num_envs, device=env.device)
        self.motion_started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.had_base_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.base_left_ground = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.early_base_fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.peak_base_height = torch.full((env.num_envs,), -math.inf, device=env.device)
        self.contact_last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.still_time = torch.zeros(env.num_envs, device=env.device)
        self.still_last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None, step=-1):
        """清除跌倒、接触、恢复与停滞历史，不写入物理状态。"""
        ids = slice(None) if env_ids is None else env_ids
        self.armed[ids] = False
        self.low_time[ids] = 0
        self.last_step[ids] = step
        self.recovering[ids] = False
        self.snake_time[ids] = 0
        self.motion_started[ids] = False
        self.had_base_contact[ids] = False
        self.base_left_ground[ids] = False
        self.early_base_fallen[ids] = False
        self.peak_base_height[ids] = -math.inf
        self.contact_last_step[ids] = step
        self.still_time[ids] = 0
        self.still_last_step[ids] = step


def fall_retry_state(env):
    if not hasattr(env, "_fall_retry_state"):
        env._fall_retry_state = FallRetryState(env)
    return env._fall_retry_state


def fallen_after_standing(env):
    """先锁存基座已升起状态，再检测持续低位，不使用足部条件门控。"""
    cfg = env.cfg
    if not (math.isfinite(cfg.fall_retry_arm_height) and math.isfinite(cfg.fall_retry_height)
            and 0 < cfg.fall_retry_height < cfg.fall_retry_arm_height):
        raise ValueError("fall retry heights must be finite and satisfy 0 < fall < arm")
    if not math.isfinite(cfg.fall_retry_hold_s) or cfg.fall_retry_hold_s <= 0:
        raise ValueError("fall_retry_hold_s must be finite and positive")
    state = fall_retry_state(env)
    fresh = state.last_step != env.common_step_counter
    height = base_height(env)
    armed = state.armed | (~state.recovering & (height >= cfg.fall_retry_arm_height))
    low_time = torch.where(armed & (height < cfg.fall_retry_height), state.low_time + env.step_dt, 0.0)
    state.armed.copy_(torch.where(fresh, armed, state.armed))
    state.low_time.copy_(torch.where(fresh, low_time, state.low_time))
    state.last_step.fill_(env.common_step_counter)
    return ~state.recovering & state.armed & (state.low_time >= cfg.fall_retry_hold_s - 1e-6)


def base_ground_contact(env):
    found = env.scene["base_ground_contact"].data.found
    return found.reshape(env.num_envs, -1).bool().any(-1)


def early_base_ground_contact(env):
    """无需此前达到高位站立，即可在早期小幅下落／接触时重置。

    初始沉降不算跌倒。指令运动开始后，观测到接触→分离→接触序列，
    无论抬升高度多少均判定跌倒。软接触始终未分离时，也检测轻微向下冲击。
    """
    cfg = env.cfg
    for name in ("early_contact_motion_threshold", "early_contact_drop_threshold", "early_contact_min_down_speed"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    state = fall_retry_state(env)
    fresh = state.contact_last_step != env.common_step_counter
    contact = base_ground_contact(env)
    height = base_height(env)
    motion = state.motion_started | (commanded_target(env).abs().amax(-1) >= cfg.early_contact_motion_threshold)
    separated = state.base_left_ground | (motion & state.had_base_contact & ~contact)
    peak = torch.maximum(state.peak_base_height, height)
    recontact = motion & separated & contact
    impact = (
        motion & contact
        & (peak - height >= cfg.early_contact_drop_threshold)
        & (base_vertical_velocity(env) <= -cfg.early_contact_min_down_speed)
    )
    fallen = state.early_base_fallen | ((recontact | impact) & ~state.recovering)
    for current, updated in (
        (state.motion_started, motion), (state.had_base_contact, state.had_base_contact | contact),
        (state.base_left_ground, separated), (state.peak_base_height, peak),
        (state.early_base_fallen, fallen),
    ):
        current.copy_(torch.where(fresh, updated, current))
    state.contact_last_step.fill_(env.common_step_counter)
    return state.early_base_fallen


def abandoned_exploration(env):
    """除非高度和关节姿态均接近目标，否则终止静止／卡住的尝试。"""
    cfg = env.cfg
    for name in ("stagnation_height_tolerance", "stagnation_joint_tolerance", "stagnation_joint_speed", "stagnation_base_speed", "stagnation_angular_speed", "stagnation_hold_s"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(cfg.stagnation_grace_s) or cfg.stagnation_grace_s < 0:
        raise ValueError("stagnation_grace_s must be finite and nonnegative")
    robot = env.scene["robot"]
    data = robot.data
    base_id = robot.find_bodies("base")[0][0]
    target = data.joint_pos.new_tensor(cfg.standing_joint_pos)
    near_goal = ((base_height(env) - cfg.standing_base_height).abs() < cfg.stagnation_height_tolerance)
    near_goal &= (data.joint_pos - target).abs().amax(-1) < cfg.stagnation_joint_tolerance
    quiet = data.joint_vel.square().mean(-1).sqrt() < cfg.stagnation_joint_speed
    quiet &= data.body_link_lin_vel_w[:, base_id].norm(dim=-1) < cfg.stagnation_base_speed
    quiet &= data.body_link_ang_vel_w[:, base_id].norm(dim=-1) < cfg.stagnation_angular_speed
    state = fall_retry_state(env)
    eligible = env.episode_length_buf * env.step_dt >= cfg.stagnation_grace_s
    stopped = eligible & ~near_goal & quiet & ~state.recovering
    fresh = state.still_last_step != env.common_step_counter
    elapsed = torch.where(stopped, state.still_time + env.step_dt, 0.0)
    state.still_time.copy_(torch.where(fresh, elapsed, state.still_time))
    state.still_last_step.fill_(env.common_step_counter)
    return state.still_time > cfg.stagnation_hold_s + 1e-6


def snake_recovery_ready(env):
    if not math.isfinite(env.cfg.recovery_target_tolerance) or env.cfg.recovery_target_tolerance <= 0:
        raise ValueError("recovery_target_tolerance must be finite and positive")
    data = env.scene["robot"].data
    return (
        snake_final_condition(env)
        & (commanded_target(env).abs().amax(-1) < env.cfg.recovery_target_tolerance)
        & (data.body_link_ang_vel_w.norm(dim=-1).amax(-1) < env.cfg.joint_speed_tolerance)
    )


def update_recovery_phase(env):
    state = fall_retry_state(env)
    fresh = state.last_step != env.common_step_counter
    fallen = fallen_after_standing(env) & fresh
    state.recovering |= fallen
    state.armed[fallen] = False
    state.low_time[fallen] = 0
    if not math.isfinite(env.cfg.recovery_snake_hold_s) or env.cfg.recovery_snake_hold_s <= 0:
        raise ValueError("recovery_snake_hold_s must be finite and positive")
    ready = state.recovering & snake_recovery_ready(env)
    held = torch.where(ready, state.snake_time + env.step_dt, 0.0)
    state.snake_time.copy_(torch.where(fresh, held, state.snake_time))
    complete = fresh & state.recovering & (state.snake_time >= env.cfg.recovery_snake_hold_s - 1e-6)
    state.recovering[complete] = False
    state.snake_time[complete] = 0
    ids = complete.nonzero(as_tuple=False).flatten()
    if ids.numel():
        # 仅处理逻辑计数器：此辅助函数不写入机器人／根部状态。
        reset_reverse_transition(env, ids)
        env.episode_length_buf[ids] = 0


def update_robust_state(env):
    update_standing_state(env)
    update_recovery_phase(env)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def recovery_time_out(env):
    return common_mdp.time_out(env) & ~fall_retry_state(env).recovering


def recovery_active_metric(env):
    return fall_retry_state(env).recovering.float()


@dataclass(kw_only=True)
class RecoveryJointTargetRateActionCfg(MirroredJointTargetRateActionCfg):
    def build(self, env):
        return RecoveryJointTargetRateAction(self, env)


class RecoveryJointTargetRateAction(MirroredJointTargetRateAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        speed = env.cfg.recovery_target_speed
        if not math.isfinite(speed) or not 0 < speed <= cfg.max_target_speed:
            raise ValueError("recovery_target_speed must be positive and within max_target_speed")
        self._recovery_speed = speed

    def process_actions(self, actions):
        previous = self.commanded_target.clone()
        super().process_actions(actions)
        step = self._recovery_speed * self._env.step_dt
        target = previous + (-previous).clamp(-step, step)
        target = self._clamp_target(target)
        recovering = fall_retry_state(self._env).recovering
        self._processed_actions.copy_(torch.where(recovering[:, None], target, self._processed_actions))


def reset_fall_retry(env, env_ids):
    randomize_initial_snake_roll(env, env_ids)
    if env.cfg.compact_standing_rewards:
        transformation_mdp.reset_transformation(env, env_ids)
    else:
        reset_reverse_transition(env, env_ids)
    apply_fixed_standing_weights(env)
    state = fall_retry_state(env)
    ids = slice(None) if env_ids is None else env_ids
    state.reset(ids, env.common_step_counter)
    if hasattr(env, "_standing_goal_valid"):
        env._standing_goal_valid[ids] = False


def disturbance_scale(env):
    cfg = env.cfg
    if not math.isfinite(cfg.disturbance_initial_scale) or not 0 <= cfg.disturbance_initial_scale <= 1:
        raise ValueError("disturbance_initial_scale must be finite and in [0, 1]")
    if cfg.disturbance_warmup_steps < 0 or cfg.disturbance_ramp_steps < 0:
        raise ValueError("disturbance warmup/ramp steps must be nonnegative")
    elapsed = env.common_step_counter - cfg.disturbance_warmup_steps
    progress = float(elapsed >= 0) if cfg.disturbance_ramp_steps == 0 else min(1.0, max(0.0, elapsed / cfg.disturbance_ramp_steps))
    return cfg.disturbance_initial_scale + (1 - cfg.disturbance_initial_scale) * progress


def disturbance_scale_metric(env):
    return torch.full((env.num_envs,), disturbance_scale(env), device=env.device)


def initial_ascent_velocity_reward(env, target_speed=0.15, pose_error_range=None):
    """仅奖励首次上升；峰值高度记忆防止奖励再次激活。"""
    threshold = env.cfg.ascent_reward_stop_height
    if not math.isfinite(threshold) or not 0 < threshold <= env.cfg.standing_base_height:
        raise ValueError("ascent_reward_stop_height must be positive and no higher than the standing goal")
    state = fall_retry_state(env)
    state.peak_base_height.copy_(torch.maximum(state.peak_base_height, base_height(env)))
    score = base_ascent_velocity_reward(env, target_speed=target_speed, pose_error_range=pose_error_range)
    return torch.where(state.peak_base_height >= threshold, 0.0, score)


def apply_fixed_standing_weights(env):
    """仅包含三个核心稠密奖励项；失败权重仍可独立调整。"""
    for name, field in (
        ("joint_standing", "standing_stage_joint_weight"),
        ("base_height_progress", "standing_stage_height_weight"),
        ("base_ascent_velocity", "standing_stage_velocity_weight"),
    ):
        env.reward_manager.get_term_cfg(name).weight = getattr(env.cfg, field)


def update_fixed_standing_state(env):
    task = transformation_mdp.state(env)
    valid = standing_final_condition(env)
    task.advance_hold(valid, env.common_step_counter, env.step_dt)
    env._standing_goal_valid = valid
    env._standing_goal_step = env.common_step_counter
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def standing_success_metric(env):
    valid = getattr(env, "_standing_goal_valid", None)
    if valid is None or getattr(env, "_standing_goal_step", -1) != env.common_step_counter:
        valid = standing_final_condition(env)
    return (transformation_mdp.state(env).hold_time >= env.cfg.success_hold_s - 1e-5) & valid


class RampedBaseImpulse(PartialBaseImpulse):
    """复用冲量引擎，并缩放幅值、清理回合重置状态。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._disabled = cfg.params.get("disturbed_fraction", 0.75) == 0 or (
            not any(cfg.params["force_range"]) and not any(cfg.params["torque_range"])
        )
        if self._disabled:
            self._disturbed_mask.zero_()

    @property
    def disturbed_mask(self):
        return self._disturbed_mask

    def __call__(self, env, env_ids, force_range, torque_range, **kwargs):
        if self._disabled:
            return
        scale = disturbance_scale(env)
        super().__call__(
            env, env_ids,
            force_range=tuple(value * scale for value in force_range),
            torque_range=tuple(value * scale for value in torque_range),
            **kwargs,
        )
        recovering = fall_retry_state(env).recovering.nonzero(as_tuple=False).flatten()
        if recovering.numel():
            self._active[recovering] = False
            self._time_remaining[recovering] = 0
            self._interval_time_left[recovering] = self._sample_cooldown(len(recovering))
            zeros = torch.zeros((len(recovering), self._num_bodies, 3), device=self._device)
            self._asset.write_external_wrench_to_sim(zeros, zeros, env_ids=recovering, body_ids=self._body_ids)

    def reset(self, env_ids=None):
        ids = torch.arange(self._num_envs, device=self._device)
        if env_ids is not None:
            ids = ids[env_ids]
        self._active[ids] = False
        self._time_remaining[ids] = 0
        self._interval_time_left[ids] = self._sample_cooldown(len(ids))
        zeros = torch.zeros((len(ids), self._num_bodies, 3), device=self._device)
        self._asset.write_external_wrench_to_sim(zeros, zeros, env_ids=ids, body_ids=self._body_ids)
