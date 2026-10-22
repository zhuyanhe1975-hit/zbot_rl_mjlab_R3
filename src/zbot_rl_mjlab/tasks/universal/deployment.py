"""综合部署学生的传感器历史输入；特权状态仅向价值网络保留。"""

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import torch
from mjlab.managers.manager_base import ManagerTermBase

from ..common import PartialBaseImpulse, constant
from ..transformation.poses import SNAKE_ROOT_POS, SNAKE_ROOT_QUAT
from ..transformation.roll_reset import collision_radial_envelope, rolled_root_states
from . import mdp
from .commands import UniversalCommand, UniversalCommandCfg


@dataclass(kw_only=True)
class DeploymentCommandCfg(UniversalCommandCfg):
    """沿用原指令参数，附加由可测状态计算的归直控制阶段。"""

    forward_speed_levels: tuple[float, ...] = (.08, .10, .12, .15, .20)
    translation_training_fraction: float = .8
    backward_training_speed: float = .05
    lateral_training_speed: float = .05
    direction_speed_multipliers: tuple[float, ...] = (1., 1.5, 2.)

    def build(self, env):
        return DeploymentCommand(self, env)


class DeploymentCommand(UniversalCommand):
    """归直阶段由编码器、内部目标和 IMU 决定，不使用高度或接触。"""

    uses_speed_course = True

    def __init__(self, cfg: DeploymentCommandCfg, env):
        super().__init__(cfg, env)
        if self.uses_speed_course:
            for direction in ('forward', 'backward', 'left', 'right'):
                levels = self.course_levels(direction)
                maximum = cfg.velocity_limits[0 if direction in ('forward', 'backward') else 1]
                if not levels or any(not math.isfinite(value) or value <= 0 for value in levels) or any(a >= b for a, b in pairwise(levels)) or levels[-1] > maximum:
                    raise ValueError('各方向速度档位必须为递增正数，且不超过对应指令上限')
            if any(not math.isfinite(value) or not 0 < value <= maximum for value, maximum in ((cfg.backward_training_speed, cfg.velocity_limits[0]), (cfg.lateral_training_speed, cfg.velocity_limits[1]))):
                raise ValueError('后退与侧移训练速度必须为正数，且不超过对应指令上限')
        if not math.isfinite(cfg.translation_training_fraction) or not 0 <= cfg.translation_training_fraction <= 1:
            raise ValueError('平移课程采样比例必须在0～1之间')
        self.motion_speed_stages = dict.fromkeys(('forward', 'backward', 'left', 'right'), 0)
        self.motion_speed_passes = self.motion_speed_stages.copy()
        self.motion_speed_failures = self.motion_speed_stages.copy()
        self.straightening_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._straightening_pending = torch.zeros_like(self.straightening_mask)

    def course_levels(self, direction):
        cfg = cast(DeploymentCommandCfg, self.cfg)
        if direction == 'forward':
            return tuple(cfg.forward_speed_levels)
        if direction not in ('backward', 'left', 'right'):
            raise ValueError('未知平移课程方向')
        base = cfg.backward_training_speed if direction == 'backward' else cfg.lateral_training_speed
        return tuple(base * multiplier for multiplier in cfg.direction_speed_multipliers)

    def set_motion_speed_stage(self, direction, stage):
        if type(stage) is not int or not 0 <= stage < len(self.course_levels(direction)):
            raise ValueError('方向速度课程档位超出范围')
        self.motion_speed_stages[direction] = stage

    @property
    def forward_speed_stage(self):
        return self.motion_speed_stages['forward']

    @forward_speed_stage.setter
    def forward_speed_stage(self, stage):
        self.set_motion_speed_stage('forward', stage)

    @property
    def forward_speed_passes(self):
        return self.motion_speed_passes['forward']

    @forward_speed_passes.setter
    def forward_speed_passes(self, passes):
        self.motion_speed_passes['forward'] = passes

    def set_forward_speed_stage(self, stage):
        self.set_motion_speed_stage('forward', stage)

    def _sample_velocity(self, ids):
        super()._sample_velocity(ids)
        if not self.cfg.training or self.pretraining:
            return
        if self.cfg.snake_teacher_enabled:
            # 第一版蛇教师只覆盖前爬；先稳定学习已有能力，再扩展方向课程。
            crawl = ids[~self.standing_mode[ids] & ~self.manual_control[ids]]
            moving = self.requested_velocity[crawl].norm(dim=-1) > 0
            self.requested_velocity[crawl] = 0
            self.requested_velocity[crawl[moving], 0] = .08
        cfg = cast(DeploymentCommandCfg, self.cfg)
        # 保留停车、其他构型和部分原采样；补充四方向均衡平移。
        eligible = self.standing_mode[ids] & ~self.manual_control[ids] & (self.requested_velocity[ids].norm(dim=-1) > 0)
        selected = ids[eligible & (torch.rand(len(ids), device=self.device) < cfg.translation_training_fraction)]
        if not len(selected):
            return
        count = len(selected)
        direction = torch.randint(4, (count,), device=self.device)
        speeds = self.requested_velocity.new_zeros(count)
        for index, name in enumerate(('forward', 'backward', 'left', 'right')):
            mask = direction == index
            size = int(mask.sum())
            if not size:
                continue
            unlocked = self.course_levels(name)[:self.motion_speed_stages[name] + 1]
            levels = speeds.new_tensor(unlocked)
            choice = torch.rand(size, device=self.device)
            previous = levels[torch.randint(len(levels), (size,), device=self.device)]
            low = min(.025 if name != 'forward' else .04, unlocked[0])
            continuous = low + torch.rand(size, device=self.device) * (levels[-1] - low)
            speeds[mask] = torch.where(choice < .5, levels[-1], torch.where(choice < .75, previous, continuous))
        self.requested_velocity[selected] = 0
        self.requested_velocity[selected, 0] = torch.where(direction == 0, speeds, torch.where(direction == 1, -speeds, 0.))
        self.requested_velocity[selected, 1] = torch.where(direction == 2, speeds, torch.where(direction == 3, -speeds, 0.))

    @property
    def observation_mode(self):
        return torch.where(self.straightening_mask, 0., self.command[:, 0])

    def _clear(self, ids):
        super()._clear(ids)
        self.straightening_mask[ids] = False
        self._straightening_pending[ids] = True

    def _update_straightening(self, ids):
        robot = self._env.scene['robot']
        joints = robot.data.joint_pos_biased
        target = self._env.action_manager.get_term('joint_pos').commanded_target
        goal = constant(joints, self._env.cfg.standing_joint_pos)
        # 重力对齐仅使用 IMU 方向，与学生的 projected_gravity 同源。
        already_standing = ((joints - goal).abs().amax(-1) < .3) & (mdp.standing.base_gravity_alignment_cost(self._env) < .45)
        # 以全长掩码表达 ids 子集，避免布尔索引逐步同步。
        chosen = torch.zeros_like(self._straightening_pending)
        chosen[ids] = True
        pending = chosen & self._straightening_pending
        mask = torch.where(pending, self.standing_mode & ~already_standing, self.straightening_mask)
        self._straightening_pending &= ~chosen
        straight = (joints.abs().amax(-1) <= .08) & (target.abs().amax(-1) <= .08)
        # 完成后保持释放；后续正常弯腿不重新触发归直。
        self.straightening_mask.copy_(torch.where(chosen, mask & ~(straight | already_standing) & self.standing_mode, mask))

    def _update_command(self, env_ids):
        self._update_straightening(self._ids(env_ids))
        super()._update_command(env_ids)


def straightening_cost(env):
    """归直阶段的小角度成本；阶段结束后归零，不以抬身替代归直。"""
    term = cast(DeploymentCommand, mdp.command_term(env))
    joints = env.scene['robot'].data.joint_pos_biased
    target = env.action_manager.get_term('joint_pos').commanded_target
    return term.straightening_mask * (joints.square() + target.square()).mean(-1)


def joint_position(env):
    """编码器角度使用传感器接口，价值网络独立读取真值。"""
    return env.scene["robot"].data.joint_pos_biased


def command(env):
    """软件构型阶段与用户速度请求，不泄漏仿真到位状态的速度门控。"""
    term = mdp.command_term(env)
    mode = getattr(term, 'observation_mode', term.command[:, 0])
    return torch.cat((mode[:, None], term.requested_velocity), -1)


def _sanitized(func):
    """NaN/Inf 置零（等价于 nan_policy="sanitize"），保留原函数名便于诊断。"""
    def observe(env):
        return torch.nan_to_num(func(env), nan=0.0, posinf=0.0, neginf=0.0)
    observe.__name__ = observe.__qualname__ = getattr(func, "__name__", "observe")
    return observe


class BentSnakeReset(ManagerTermBase):
    """保留一半规范蛇形，另一半使用带弯角初态；缓存碰撞包络避免逐步编译。"""

    def __init__(self, cfg, env):
        super().__init__(env)
        params = cfg.params
        fraction, limit, roll = (params[key] for key in ('bend_fraction', 'bend_limit', 'roll_limit_deg'))
        if not all(math.isfinite(value) for value in (fraction, limit, roll)) or not (0 <= fraction <= 1 and 0 < limit <= .5 and 0 <= roll <= 90):
            raise ValueError('弯曲初态需要合法比例、0～0.5弧度关节范围及0～90度侧滚范围')
        self._bank = None

    def __call__(self, env, env_ids, *, bend_fraction, bend_limit, roll_limit_deg):
        mdp.reset_universal(env, env_ids)
        if not env.cfg.commands['transform'].training:
            return
        robot = env.scene['robot']
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        # 基础重置已决定双足/蛇形；只改写其中仍为零关节的蛇形。
        snake = robot.data.joint_pos[ids].abs().amax(-1) < 1e-6
        if getattr(env.cfg, 'snake_teacher_enabled', False):
            command = mdp.command_term(env)
            crawl = (ids % 5 == 4) if command.pretraining else ~command.standing_mode[ids]
            snake &= ~crawl
        selected = ids[snake & (torch.rand(len(ids), device=env.device) < bend_fraction)]
        if not len(selected):
            return
        reference = robot.data.default_root_state
        if self._bank is None:
            # 独立种子只定义姿态库，不改变训练随机数；采样与侧滚仍使用训练种子。
            generator = torch.Generator().manual_seed(20261018)
            poses = (torch.rand(32, 6, generator=generator) * 2 - 1) * bend_limit
            self._bank = []
            for pose in poses:
                radial, low, high = collision_radial_envelope(tuple(pose.tolist()))
                self._bank.append((pose.to(reference), reference.new_tensor(radial), low, high))
        assignments = torch.randint(len(self._bank), (len(selected),), device=env.device)
        for index, (pose, radial, low, high) in enumerate(self._bank):
            chosen = selected[assignments == index]
            if not len(chosen):
                continue
            root = reference[chosen].clone()
            root[:, :3] = root.new_tensor(SNAKE_ROOT_POS)
            root[:, 3:7] = root.new_tensor(SNAKE_ROOT_QUAT)
            angles = (torch.rand(len(chosen), device=env.device) * 2 - 1) * math.radians(roll_limit_deg)
            root = rolled_root_states(root, env.scene.env_origins[chosen], angles, radial, low, high, .003)
            joints = pose.expand(len(chosen), -1)
            robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=chosen)
            robot.write_root_state_to_sim(root, env_ids=chosen)


class StudentBaseImpulse(PartialBaseImpulse):
    """教师与学生共享同一受扰环境，按学生实际 PPO 进度增加外扰。"""

    def __call__(self, env, env_ids, *args, pretraining_scale=.2, ramp_iterations=500, **kwargs):
        training = env.cfg.commands['transform'].training
        if training:
            elapsed = 0 if getattr(env, '_student_pretraining', True) else getattr(env, '_student_rl_iteration', 0)
            fraction = min(1., max(0., elapsed / ramp_iterations))
            scale = pretraining_scale + (1 - pretraining_scale) * fraction
        else:
            # 鲁棒验收及回放使用所配置的完整外扰，标称验收会移除此事件。
            scale = 1.
        kwargs['force_range'] = tuple(value * scale for value in kwargs['force_range'])
        kwargs['torque_range'] = tuple(value * scale for value in kwargs['torque_range'])
        return super().__call__(env, env_ids, *args, **kwargs)

