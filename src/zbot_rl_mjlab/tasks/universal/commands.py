"""综合运动指令：构型切换期间平滑停车，到位后恢复待执行速度。"""
import math
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class UniversalCommandCfg(CommandTermCfg):
    training: bool = True
    mode: str = "source"
    auto_switch_on_success: bool = False
    velocity_limits: tuple[float, float, float] = (.25, .20, .8)
    hold_s: float = .15
    smoothing_s: float = .25
    transition_timeout_s: float = 8.
    gait_frequency: float = .5

    def build(self, env) -> "UniversalCommand":
        return UniversalCommand(self, env)


class UniversalCommand(CommandTerm):
    """四阶段共享策略；命令变化不修改机器人状态或动作积分目标。"""
    _env: Any
    cfg: UniversalCommandCfg

    def __init__(self, cfg: UniversalCommandCfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.gait_frequency) or not .25 <= cfg.gait_frequency <= 1.:
            raise ValueError("步态频率必须在 0.25 至 1 Hz 范围内")
        if cfg.mode not in ("source", "stand", "snake"):
            raise ValueError("构型模式必须为 source、stand 或 snake")
        if cfg.auto_switch_on_success:
            raise ValueError("综合任务不支持到位后自动反向")
        if not all(math.isfinite(v) and v > 0 for v in (cfg.hold_s, cfg.smoothing_s)):
            raise ValueError("保持时间和平滑时间必须为正有限值")
        if not math.isfinite(cfg.transition_timeout_s) or cfg.transition_timeout_s <= 0:
            raise ValueError("变形期限必须为正有限值")
        if not all(math.isfinite(v) and v > 0 for v in cfg.velocity_limits):
            raise ValueError("速度范围必须为正有限值")
        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        self._command[:, 0] = 1
        self.requested_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.manual_control = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.transition_mask = torch.ones_like(self.manual_control)
        self.standing_achieved = torch.zeros_like(self.manual_control)
        self.phase = torch.full((self.num_envs,), 3, dtype=torch.long, device=self.device)
        self.hold_time = torch.zeros(self.num_envs, device=self.device)
        self.start_step = torch.zeros_like(self.phase)
        self.last_step = torch.full_like(self.phase, -1)
        self.heading = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading[:, 0] = 1
        self.initial_heading = self.heading.clone()
        self.heading_initialized = torch.zeros_like(self.manual_control)
        self.pretraining = False
        self.tracking_ema = torch.zeros(2, device=self.device)
        self.potential = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.potential_step = torch.full_like(self.phase, -1)
        self.progress_reward = torch.zeros(self.num_envs, device=self.device)
        self.gait_frequency = torch.full((self.num_envs,), cfg.gait_frequency, device=self.device)

    @property
    def command(self):
        return self._command

    @property
    def velocity_command(self):
        return self._command[:, 1:]

    @property
    def standing_mode(self):
        return self._command[:, 0] > 0

    def _ids(self, ids):
        return torch.arange(self.num_envs, device=self.device) if ids is None else ids

    def set_pretraining(self, enabled: bool):
        """只选择下一次重置的监督场景，不修改当前物理状态。"""
        self.pretraining = bool(enabled)

    def _clear(self, ids):
        self.potential[ids] = float("nan")
        self.potential_step[ids] = -1
        self.hold_time[ids] = 0
        self.standing_achieved[ids] = False
        self.last_step[ids] = -1
        self.start_step[ids] = self._env.common_step_counter
        self.transition_mask[ids] = True
        self.phase[ids] = torch.where(self.standing_mode[ids], 3, 1)

    def reset(self, env_ids):
        ids = self._ids(env_ids)
        self.manual_control[ids] = False
        self.requested_velocity[ids] = 0
        self.velocity_command[ids] = 0
        self.heading[ids] = self.heading.new_tensor([1., 0.])
        self.initial_heading[ids] = self.heading[ids]
        self.heading_initialized[ids] = False
        self._clear(ids)
        return super().reset(ids)

    def _resample_command(self, env_ids):
        ids = env_ids[~self.manual_control[env_ids]]
        if not len(ids):
            return
        if self.pretraining and self.cfg.training:
            group = ids % 4
            previous_mode = self._command[ids, 0].clone()
            self._command[ids, 0] = torch.where(group == 2, -1., 1.)
            self.requested_velocity[ids] = 0
            self.requested_velocity[ids[group == 0], 0] = .08
            self.requested_velocity[ids[group == 1], 1] = .05
            self._clear(ids[self._command[ids, 0] != previous_mode])
            return
        # 未完成的变形必须到位或超时，不能用自动换目标逃避失败。
        ids = ids[~self.transition_mask[ids] | (self.start_step[ids] == self._env.common_step_counter)]
        if not len(ids):
            return
        previous_mode = self._command[ids, 0].clone()
        if self.cfg.training:
            if self.cfg.mode != "source":
                choose_stand = torch.full((len(ids),), self.cfg.mode == "stand", device=self.device, dtype=torch.bool)
            elif self.num_envs == 1:
                choose_stand = torch.rand(len(ids), device=self.device) < .5
            else:
                # 用剩余有效目标补足名额，避免短失败回合挤掉较难技能。
                current = self.standing_mode
                remaining = current.sum() - current[ids].sum()
                vacancies = (int(self.num_envs * .5 + .5) - remaining).clamp(0, len(ids))
                choose_stand = torch.randperm(len(ids), device=self.device) < vacancies
            self._command[ids, 0] = torch.where(choose_stand, 1., -1.)
            self._sample_velocity(ids)
        else:
            self._command[ids, 0] = -1. if self.cfg.mode == "snake" else 1.
            self.requested_velocity[ids] = 0
            self.time_left[ids] = math.inf
        self._clear(ids[self._command[ids, 0] != previous_mode])

    def _sample_velocity(self, ids):
        """默认速度课程；部署实验通过继承扩展采样，不复制构型调度。"""
        limits = self.requested_velocity.new_tensor(self.cfg.velocity_limits)
        competence = self.tracking_ema[self.standing_mode[ids].long()]
        scale = .2 + .8 * ((competence - .6) / .25).clamp(0, 1)
        sampled = (torch.rand(len(ids), 3, device=self.device) * 2 - 1) * limits * scale[:, None]
        # 先分别学会前后、左右及转向，达标后再采样组合指令。
        axis = torch.randint(3, (len(ids),), device=self.device)
        single_axis = torch.nn.functional.one_hot(axis, 3).bool()
        self.requested_velocity[ids] = torch.where((competence >= .75)[:, None] | single_axis, sampled, 0.)
        quiet = torch.rand(len(ids), device=self.device) < .25
        self.requested_velocity[ids[quiet]] = 0

    def _update_metrics(self):
        pass

    def _update_command(self, env_ids):
        from .mdp import configuration_ready, motion_tracking_score, tracking_velocity

        ids = self._ids(env_ids)
        if env_ids is not None:
            return
        fresh = self.last_step[ids] != self._env.common_step_counter
        if self.cfg.training and not self.pretraining and fresh.any():
            scores = motion_tracking_score(tracking_velocity(self._env), self.velocity_command, self.cfg.velocity_limits)
            moving = ~self.transition_mask & (self.velocity_command.norm(dim=-1) > .01) & fresh
            for mode in (0, 1):
                selected = moving & (self.standing_mode == bool(mode))
                if selected.any():
                    self.tracking_ema[mode].lerp_(scores[selected].mean().clamp(0, 1), .01)
        ready = configuration_ready(self._env)
        relaxed = configuration_ready(self._env, relaxed=True)
        held = torch.where(ready, self.hold_time + self._env.step_dt, 0.)
        self.hold_time[ids] = torch.where(fresh, held[ids], self.hold_time[ids])
        settled = self.hold_time >= self.cfg.hold_s
        # 已进入运动阶段后使用更宽阈值，避免合法摆动导致频繁切换。
        was_transition = self.transition_mask.clone()
        self.transition_mask[ids] = torch.where(self.transition_mask[ids], ~settled[ids], ~relaxed[ids])
        entered = ids[~was_transition[ids] & self.transition_mask[ids]]
        self.start_step[entered] = self._env.common_step_counter
        self.phase[ids] = torch.where(self.standing_mode[ids], torch.where(self.transition_mask[ids], 3, 0), torch.where(self.transition_mask[ids], 1, 2))
        self.standing_achieved[ids] |= self.phase[ids] == 0
        target = torch.where(self.transition_mask[:, None], 0., self.requested_velocity)
        alpha = min(1., self._env.step_dt / self.cfg.smoothing_s)
        updated = self.velocity_command + alpha * (target - self.velocity_command)
        self.velocity_command[ids] = torch.where(fresh[:, None], updated[ids], self.velocity_command[ids])
        self.last_step[ids] = self._env.common_step_counter

    def set_target(self, target, env_ids=None):
        if target not in ("stand", "snake"):
            raise ValueError("目标必须为 stand 或 snake")
        ids = self._ids(env_ids)
        value = 1. if target == "stand" else -1.
        changed = ids[self._command[ids, 0] != value]
        self._command[ids, 0] = value
        self._clear(changed)
        self.manual_control[ids] = True
        self.time_left[ids] = math.inf

    def set_velocity(self, vx, vy, wz, env_ids=None):
        values = (vx, vy, wz)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("速度指令必须为有限值")
        ids = self._ids(env_ids)
        limits = self.requested_velocity.new_tensor(self.cfg.velocity_limits)
        self.requested_velocity[ids] = self.requested_velocity.new_tensor(values).clamp(-limits, limits)
        self.manual_control[ids] = True
        self.time_left[ids] = math.inf

    def stop(self, env_ids=None):
        self.set_velocity(0., 0., 0., env_ids)
