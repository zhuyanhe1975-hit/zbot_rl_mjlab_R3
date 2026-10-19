"""独立蛇形教师的平滑全向速度指令与手动控制。"""

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class SnakeVelocityCommandCfg(CommandTermCfg):
    training: bool = True
    sample_velocity_commands: bool = True
    velocity_limits: tuple[float, float, float] = (.25, .25, math.pi / 2)
    smoothing_s: float = .3
    frequency: float = .7
    translation_probability: float = .75
    turn_probability: float = .10
    stop_probability: float = .10
    translation_min_speed: float = 0.
    translation_curriculum: tuple[float, ...] = ()

    def build(self, env) -> "SnakeVelocityCommand":
        return SnakeVelocityCommand(self, env)


class SnakeVelocityCommand(CommandTerm):
    """从第一轮训练覆盖六个方向；按键只改变指令，不修改物理状态。"""

    _env: Any
    cfg: SnakeVelocityCommandCfg

    def __init__(self, cfg: SnakeVelocityCommandCfg, env):
        super().__init__(cfg, env)
        if len(cfg.velocity_limits) != 3 or not all(
            math.isfinite(value) and value > 0 for value in cfg.velocity_limits
        ):
            raise ValueError("蛇形速度上限必须为三个正有限值")
        if not math.isfinite(cfg.smoothing_s) or cfg.smoothing_s <= 0:
            raise ValueError("蛇形指令平滑时间必须为正有限值")
        if not math.isfinite(cfg.frequency) or cfg.frequency <= 0:
            raise ValueError("蛇形步态频率必须为正有限值")
        probabilities = (cfg.translation_probability, cfg.turn_probability, cfg.stop_probability)
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or sum(probabilities) > 1:
            raise ValueError("平移、转向和停车概率须在[0,1]且总和不超过1")
        caps = cfg.translation_curriculum
        if (any(not math.isfinite(cap) or not 0 < cap <= min(cfg.velocity_limits[:2]) for cap in caps)
                or any(a >= b for a, b in pairwise(caps))):
            raise ValueError("平移课程上限须为递增正有限值，且不超过平移速度上限")
        initial_cap = caps[0] if caps else min(cfg.velocity_limits[:2])
        if not math.isfinite(cfg.translation_min_speed) or not 0 <= cfg.translation_min_speed <= initial_cap:
            raise ValueError("最小平移速度须在零与初始平移上限之间")
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.requested_velocity = torch.zeros_like(self._command)
        self.manual_control = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gait_frequency = torch.full((self.num_envs,), cfg.frequency, device=self.device)
        self.translation_stage = 0
        self._translation_success = torch.zeros(4, device=self.device)
        self._translation_observed_s = torch.zeros(4, device=self.device)
        self._last_curriculum_step = -1

    @property
    def current_translation_cap(self):
        caps = self.cfg.translation_curriculum
        return caps[self.translation_stage] if caps and self.cfg.training else min(self.cfg.velocity_limits[:2])

    def curriculum_state(self):
        return {"caps": self.cfg.translation_curriculum,
                "min_speed": self.cfg.translation_min_speed, "stage": self.translation_stage}

    def validate_curriculum_state(self, state):
        if not isinstance(state, dict):
            raise TypeError("蛇形平移课程检查点格式无效")
        stage = state.get("stage")
        if (tuple(state.get("caps", ())) != self.cfg.translation_curriculum
                or state.get("min_speed") != self.cfg.translation_min_speed
                or type(stage) is not int
                or not 0 <= stage < max(1, len(self.cfg.translation_curriculum))):
            raise ValueError("蛇形平移课程检查点与当前课程配置不兼容")
        return stage

    def load_curriculum_state(self, state):
        self.translation_stage = self.validate_curriculum_state(state)
        self._translation_success.zero_()
        self._translation_observed_s.zero_()
        self._last_curriculum_step = -1

    @property
    def command(self):
        return self._command

    @property
    def velocity_command(self):
        return self._command

    def _ids(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)
        return ids if env_ids is None else ids[env_ids]

    def reset(self, env_ids):
        ids = self._ids(env_ids)
        self.manual_control[ids] = False
        self.requested_velocity[ids] = 0
        self._command[ids] = 0
        return super().reset(ids)

    def _resample_command(self, env_ids):
        ids = env_ids[~self.manual_control[env_ids]]
        if not len(ids):
            return
        if not self.cfg.training or not self.cfg.sample_velocity_commands:
            self.requested_velocity[ids] = 0
            self.time_left[ids] = math.inf
            return
        count = len(ids)
        limits = self.requested_velocity.new_tensor(self.cfg.velocity_limits)
        if self.cfg.training and self.cfg.translation_curriculum:
            limits[:2].clamp_(max=self.current_translation_cap)
        # 按配置比例采样，平移幅值受当前课程阶段限制。
        kind = torch.rand(count, device=self.device)
        translation_end = self.cfg.translation_probability
        turn_end = translation_end + self.cfg.turn_probability
        stop_end = turn_end + self.cfg.stop_probability
        translating = kind < translation_end
        turning = (kind >= translation_end) & (kind < turn_end)
        amplitude = torch.rand(count, device=self.device)
        axis = torch.where(turning, 2, torch.randint(2, (count,), device=self.device))
        signs = torch.where(torch.rand(count, device=self.device) < .5, 1., -1.)
        sampled = (torch.rand(count, 3, device=self.device) * 2 - 1) * limits
        minimum = self.cfg.translation_min_speed
        if minimum:
            unit = sampled[:, :2] / limits[:2]
            sampled[:, :2] = torch.where(unit >= 0, 1., -1.) * (minimum + unit.abs() * (limits[:2] - minimum))
        single = torch.zeros_like(sampled)
        magnitude = torch.where(turning, amplitude * limits[axis], minimum + amplitude * (limits[axis] - minimum))
        single[torch.arange(count, device=self.device), axis] = signs * magnitude
        sampled = torch.where((translating | turning)[:, None], single, sampled)
        self.requested_velocity[ids] = torch.where(
            ((kind >= turn_end) & (kind < stop_end))[:, None], 0., sampled
        )

    def _update_metrics(self):
        pass

    def _debug_vis_impl(self, visualizer):
        """运动site原点显示平滑后的指令：蓝色平移、橙色偏航（向上为正）。"""
        from .mdp import motion_axes, motion_site_index, world_com_velocity

        indices = visualizer.get_env_indices(self.num_envs)
        if not indices:
            return
        origins = self._env.scene["robot"].data.site_pos_w[:, motion_site_index(self._env)]
        if getattr(self._env.cfg, "observation_reference", "command") == "world_x":
            reference = torch.zeros_like(origins)
            reference[:, :2] = motion_axes(self._env)[0]
            vectors_and_colors = (
                (reference, (.2, .4, 1., .9)),
                (world_com_velocity(self._env) / .25, (.1, 1., .2, .9)),
            )
        else:
            forward, left = motion_axes(self._env)
            normalized = self.command / self.command.new_tensor(self.cfg.velocity_limits)
            linear = torch.zeros_like(origins)
            linear[:, :2] = forward * normalized[:, :1] + left * normalized[:, 1:2]
            yaw = torch.zeros_like(origins)
            yaw[:, 2] = normalized[:, 2]
            vectors_and_colors = ((linear, (.2, .4, 1., .9)), (yaw, (1., .55, .1, .9)))
        starts = origins.detach().cpu().numpy()
        # 回放以零指令启动；保留基点，避免把静止误认为可视化未开启。
        for i in indices:
            visualizer.add_sphere(starts[i], radius=.012, color=(.2, .4, 1., .9))
        for vectors, color in vectors_and_colors:
            norms = vectors.norm(dim=-1, keepdim=True)
            # 单次按键的速度很小，箭头至少显示4厘米；方向和零值保持真实。
            lengths = (.15 * norms).clamp_min(.04)
            ends = (origins + vectors / norms.clamp_min(1e-6) * lengths).detach().cpu().numpy()
            visible = (norms[:, 0] > 1e-3).cpu().tolist()
            for i in indices:
                if visible[i]:
                    visualizer.add_arrow(starts[i], ends[i], color=color, width=.008)

    def _update_command(self, env_ids):
        # 重置路径只清零，避免局部重置时多推进一次其他环境的滤波器。
        if env_ids is not None:
            return
        self._update_translation_curriculum()
        alpha = 1 - math.exp(-self._env.step_dt / self.cfg.smoothing_s)
        self._command.lerp_(self.requested_velocity, alpha)

    def _update_translation_curriculum(self):
        """仅四向纯平移的稳定跟踪可升级；停车、转向、手动控制均不参与。"""
        if (not self.cfg.training or not self.cfg.translation_curriculum
                or self.translation_stage >= len(self.cfg.translation_curriculum) - 1):
            return
        step = self._env.common_step_counter
        if step == self._last_curriculum_step:
            return
        self._last_curriculum_step = step
        from .mdp import head_tilt_limit_reached, planar_velocity

        requested = self.requested_velocity
        planar = requested[:, :2]
        magnitude = planar.norm(dim=-1)
        selected = ((planar.abs() > .001).sum(-1) == 1) & (requested[:, 2].abs() <= .001)
        selected &= ~self.manual_control & (magnitude >= .8 * self.current_translation_cap)
        selected &= ((self.command[:, :2] - planar).norm(dim=-1) <= .1 * magnitude)
        selected &= self.command[:, 2].abs() <= .001
        actual = planar_velocity(self._env)
        success = ((actual[:, :2] - self.command[:, :2]).norm(dim=-1) <= .3 * magnitude)
        success &= actual[:, 2].abs() <= .3
        success &= ~head_tilt_limit_reached(self._env)
        axis = planar.abs().argmax(-1)
        component = planar.gather(1, axis[:, None])[:, 0]
        direction = 2 * axis + (component < 0).long()
        counts = torch.zeros_like(self._translation_success).scatter_add_(0, direction, selected.float())
        successes = torch.zeros_like(counts).scatter_add_(0, direction, (selected & success).float())
        present = counts > 0
        dt = self._env.step_dt
        alpha = 1 - math.exp(-dt / 5.)
        scores = successes / counts.clamp_min(1)
        self._translation_success.lerp_(torch.where(present, scores, self._translation_success), alpha)
        self._translation_observed_s += present * dt
        # 每秒检查一次，避免每个GPU仿真步为全局升级判定同步到CPU。
        interval = max(1, round(1 / dt))
        if step % interval == 0 and ((self._translation_observed_s >= 10.) & (self._translation_success >= .8)).all():
            self.translation_stage += 1
            self._translation_success.zero_()
            self._translation_observed_s.zero_()

    def set_velocity(self, vx, vy, wz, env_ids=None):
        values = (vx, vy, wz)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("蛇形速度指令必须为有限值")
        ids = self._ids(env_ids)
        limits = self.requested_velocity.new_tensor(self.cfg.velocity_limits)
        self.requested_velocity[ids] = self.requested_velocity.new_tensor(values).clamp(-limits, limits)
        self.manual_control[ids] = True
        self.time_left[ids] = math.inf

    def stop(self, env_ids=None):
        self.set_velocity(0., 0., 0., env_ids)
