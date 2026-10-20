"""冻结多教师及无副作用观测适配。

位置教师通过单步限速转换指导目标速度策略；执行器接口不同，不能精确复现旧步态。
蛇形教师仅在显式启用且处于归档验证覆盖范围时指导世界+X推进。
"""

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import torch
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from .robot import JOINT_NAMES
from .tasks.snake import mdp as snake_mdp
from .tasks.transformation.actions import expand_mirror, project_mirror
from .tasks.transformation.teacher_observations import (
    teacher_reference_observation_groups,
)
from .tasks.walking import mdp as walking_mdp
from .tasks.walking.env_cfg import ZbotEnvCfg

_ROOT = Path(__file__).resolve().parents[2]
_TEACHERS = {
    "walking": ("Velocity-Energy-Disturbed-Walking", "velocity_energy_disturbed_2026-10-10_iter499/model_499.pt", 32, 6),
    "omni": ("Omni-Velocity-Energy-Walking", "omni_velocity_energy_2026-10-10_iter499/model_499.pt", 33, 6),
    "transform": ("Transform-Snake-Biped-Bidirectional", "transform_snake_biped_bidirectional_2026-10-17_iter599/model_599.pt", 37, 3),
}


_SNAKE_SPEC = ("Snake-Omni-Teacher", "snake_world_x_2026-10-19_iter1499/model_1499.pt", 51, 6)
SNAKE_CHECKPOINT = str(_ROOT / "bests" / _SNAKE_SPEC[1])


DEFAULT_CHECKPOINTS = {key: str(_ROOT / "bests" / spec[1]) for key, spec in _TEACHERS.items()}


def validate_teacher_selection(names: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """None retains legacy defaults; an explicit sequence selects exactly its entries."""
    if names is None:
        return None
    if not isinstance(names, (tuple, list)) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError("教师选择必须是非空名称序列")
    if len(set(names)) != len(names):
        raise ValueError("教师选择不能重复")
    unknown = set(names) - {*_TEACHERS, "snake"}
    if unknown:
        raise ValueError(f"未知教师：{sorted(unknown)}")
    return tuple(names)


@dataclass
class TeacherTargets:
    actions: torch.Tensor
    weights: torch.Tensor
    counts: dict[str, int]
    teacher_ids: torch.Tensor | None = None


def _joint_values(value, reference):
    """按真实关节顺序展开旧配置，禁止静默接受未知缩放规则。"""
    if isinstance(value, dict):
        import re

        result = torch.zeros(6, device=reference.device, dtype=reference.dtype)
        matched = torch.zeros(6, dtype=torch.bool, device=reference.device)
        for pattern, number in value.items():
            for index, name in enumerate(JOINT_NAMES):
                if re.fullmatch(pattern, name):
                    result[index] = number
                    matched[index] = True
        if not matched.all():
            raise ValueError("教师关节配置未覆盖全部六个关节")
        return result
    return reference.new_full((6,), value)


def position_to_rate(position, target, limits, dt, max_speed):
    """将绝对位置目标投影为学生一步可实现的归一化目标速度。"""
    if dt <= 0 or max_speed <= 0:
        raise ValueError("控制周期和最大目标速度必须为正")
    desired = position.clamp(min=limits[..., 0], max=limits[..., 1])
    return ((desired - target) / (dt * max_speed)).clamp(-1, 1)


class TeacherEnsemble:
    """一个学生环境共享冻结网络，不创建教师仿真环境。"""

    def __init__(self, env, device=None, checkpoints: dict[str, str] | None = None, enabled_teachers: tuple[str, ...] | None = None):
        self.env = env
        self.device = torch.device(device or env.device)
        if self.device != torch.device(env.device):
            raise ValueError("教师与学生必须在同一设备，避免逐步跨设备复制")
        checkpoints = checkpoints or {}
        all_specs = {**_TEACHERS, "snake": _SNAKE_SPEC}
        names = validate_teacher_selection(enabled_teachers)
        unknown = checkpoints.keys() - all_specs.keys()
        if unknown:
            raise ValueError(f"未知教师：{sorted(unknown)}")
        if names is None:
            names = tuple(_TEACHERS) + (("snake",) if "snake" in checkpoints else ())
        disabled = checkpoints.keys() - set(names)
        if disabled:
            raise ValueError(f"检查点覆盖包含未启用教师：{sorted(disabled)}")
        specs = {key: all_specs[key] for key in names}
        self.models = {}
        self.configs = {}
        self.defaults = {}
        reference = env.scene["robot"].data.joint_pos
        self._frequency = {}
        self._initial_forward = {}
        self._last_episode = torch.full((env.num_envs,), -1, device=self.device)
        self._last_stamp = None
        self._straighten_stamp = None
        self._straighten_episode = torch.full_like(self._last_episode, -1)
        self._straighten_standing = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
        self._straighten_active = torch.zeros_like(self._straighten_standing)
        # 先检查全部路径，避免漏掉教师时只加载了一部分模型。
        paths = {}
        for key in specs:
            path = Path(checkpoints[key]).expanduser() if key in checkpoints else Path(SNAKE_CHECKPOINT if key == "snake" else DEFAULT_CHECKPOINTS[key])
            if not path.is_absolute():
                path = _ROOT / path
            if not path.is_file():
                raise FileNotFoundError(f"{key} 教师检查点不存在：{path}")
            paths[key] = path
        for key, (suffix, _, dimension, outputs) in specs.items():
            task = "Mjlab-Zbot-6dof-" + suffix
            cfg = load_env_cfg(task, play=True)
            actor_cfg = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(task)).actor
            checkpoint = torch.load(paths[key], map_location=self.device, weights_only=False)
            if key == "snake":
                self._restore_snake_recipe(cfg, checkpoint)
            obs = TensorDict({"actor": reference.new_zeros(env.num_envs, dimension)}, batch_size=[env.num_envs])
            model = MLPModel(obs=obs, obs_groups={"actor": ["actor"]}, obs_set="actor", output_dim=outputs, hidden_dims=actor_cfg.hidden_dims, activation=actor_cfg.activation, obs_normalization=actor_cfg.obs_normalization, distribution_cfg=actor_cfg.distribution_cfg).to(self.device)
            model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
            model.requires_grad_(False)
            model.eval()
            self.models[key] = model
            self.configs[key] = cfg
            self.defaults[key] = _joint_values(cfg.scene.entities["robot"].init_state.joint_pos, reference)
            if key in ("walking", "omni"):
                # 数据代理缺少公开步态频率时才使用备援值；真实综合任务共享学生条件。
                walking_cfg = cast(ZbotEnvCfg, cfg)
                self._frequency[key] = torch.linspace(walking_cfg.step_frequency_min, walking_cfg.step_frequency_max, env.num_envs, device=self.device)

    @property
    def teacher_names(self):
        return tuple(self.models)

    @staticmethod
    def _restore_snake_recipe(cfg, checkpoint):
        # 检查完整归档语义，禁止把同维度但不同配方的网络静默当成该教师。
        from .experiments.snake import SnakeTeacherRunner

        recipe = SnakeTeacherRunner._describe_checkpoint(checkpoint)
        for field, source in (("motion_frame", "motion_frame"), ("head_body", "motion_body"),
                              ("motion_site", "motion_site"), ("tail_site", "tail_site"),
                              ("observation_reference", "observation_reference"),
                              ("head_gyro_frame", "head_gyro_frame"),
                              ("head_yaw_limit", "head_yaw_limit"), ("body_yaw_limit", "body_yaw_limit"),
                              ("head_yaw_integral_time_scale", "head_yaw_integral_time_scale"),
                              ("body_yaw_integral_time_scale", "body_yaw_integral_time_scale")):
            setattr(cfg, field, recipe[source])
        # 第1499轮归档的其余观测常数不随当前训练默认值漂移。
        cfg.commands["motion"].frequency = .7
        cfg.head_up_axis = (1., 0., 0.)
        cfg.head_forward_axis = (0., 0., -1.)
        cfg.head_tilt_limit = torch.pi / 3
        cfg.actions["joint_pos"].target_limit = recipe["joint_target_limit"]
        cfg.actions["joint_pos"].max_target_speed = recipe["max_target_speed"]

    def _snake_observation(self):
        """旧蛇形状态函数的所有历史仅写入独立代理，学生缓存不受影响。"""
        env = self.env
        action = env.action_manager.get_term("joint_pos")
        cfg = self.configs["snake"]
        if not hasattr(self, "_snake_proxy"):
            self._snake_proxy = SimpleNamespace()
            self._snake_episode = torch.full_like(env.episode_length_buf, -1)
            self._snake_stamp = None
        proxy = self._snake_proxy
        proxy.cfg, proxy.scene, proxy.sim = cfg, env.scene, env.sim
        proxy.device, proxy.num_envs = env.device, env.num_envs
        proxy.step_dt, proxy.common_step_counter = env.step_dt, env.common_step_counter
        proxy.episode_length_buf = env.episode_length_buf
        native_action = SimpleNamespace(
            commanded_target=action.commanded_target,
            raw_action=(action.raw_action * action.rate_cfg.max_target_speed / cfg.actions["joint_pos"].max_target_speed).clamp(-1, 1),
        )
        proxy.action_manager = SimpleNamespace(get_term=lambda name: native_action)
        proxy.command_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(cfg=cfg.commands["motion"]))
        stamp = (env.common_step_counter, getattr(env, "_observation_generation", 0))
        if stamp != self._snake_stamp:
            fresh = (env.episode_length_buf <= self._snake_episode) | (self._snake_episode < 0)
            if fresh.any():
                heading = snake_mdp.head_forward(proxy).clone()
                if not hasattr(proxy, "_snake_initial_yaw"):
                    proxy._snake_initial_yaw = torch.zeros(env.num_envs, device=env.device)
                proxy._snake_initial_yaw[fresh] = torch.atan2(heading[fresh, 1], heading[fresh, 0])
                ids = fresh.nonzero(as_tuple=False).flatten()
                snake_mdp.body_yaw_integral_state(proxy).reset(ids, env.common_step_counter)
                snake_mdp.head_yaw_integral_state(proxy).reset(ids, env.common_step_counter)
            self._snake_episode.copy_(env.episode_length_buf)
            self._snake_stamp = stamp
        return snake_mdp.actor_state(proxy)

    def _update_memory(self):
        env = self.env
        stamp = (env.common_step_counter, getattr(env, "_observation_generation", 0))
        if stamp == self._last_stamp:
            return
        episode = env.episode_length_buf
        fresh = (episode <= self._last_episode) | (self._last_episode < 0)
        forward = walking_mdp._forward_world(env)
        for key in ("walking", "omni"):
            initial = self._initial_forward.get(key, forward.clone())
            self._initial_forward[key] = torch.where(fresh[:, None], forward, initial)
        self._last_episode.copy_(episode)
        self._last_stamp = stamp

    def _straightening_mask(self, height):
        """每次低位上升仅归直一次；归直成功后弯腿起立不得重新触发。"""
        if not getattr(self.env.cfg, "straighten_before_stand", False):
            return torch.zeros_like(self._straighten_active)
        env = self.env
        command = env.command_manager.get_term("transform")
        # 部署学生与教师共用由可测状态驱动的软件阶段，禁止另以特权高度分流。
        if hasattr(command, "straightening_mask"):
            return command.straightening_mask
        stamp = (env.common_step_counter, getattr(env, "_observation_generation", 0))
        fresh = torch.zeros_like(self._straighten_active)
        if stamp != self._straighten_stamp:
            fresh = (env.episode_length_buf <= self._straighten_episode) | (self._straighten_episode < 0)
            self._straighten_episode.copy_(env.episode_length_buf)
            self._straighten_stamp = stamp
        new_ascent = fresh | (command.standing_mode & ~self._straighten_standing)
        self._straighten_active[new_ascent] = (command.standing_mode & command.transition_mask & (height < .14))[new_ascent]
        self._straighten_active &= command.standing_mode & command.transition_mask
        action = env.action_manager.get_term("joint_pos")
        straight = (action.commanded_target.abs().amax(-1) <= .08) & (env.scene["robot"].data.joint_pos.abs().amax(-1) <= .08)
        self._straighten_active &= ~straight
        self._straighten_standing.copy_(command.standing_mode)
        return self._straighten_active

    def _walking_velocity_command(self):
        """先恢复世界平面速度，再投影到旧教师前／左轴，避免朝向约定漂移。"""
        env = self.env
        command = env.command_manager.get_term("transform")
        heading = getattr(command, "heading", None)
        if heading is None:
            # 纯数据测试代理未提供朝向时，约定已使用旧行走坐标系。
            return command.velocity_command
        lateral = torch.stack((-heading[:, 1], heading[:, 0]), -1)
        world = heading * command.velocity_command[:, :1] + lateral * command.velocity_command[:, 1:2]
        forward = walking_mdp._forward_world(env)[:, :2]
        left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
        return torch.stack(((world * forward).sum(-1), (world * left).sum(-1), command.velocity_command[:, 2]), -1)

    def _walking_observation_state(self, velocity=None, gravity=None):
        # 仅在单次查询中共享；公开 observation 每次仍读取当前环境状态。
        return {
            "current": walking_mdp._forward_world(self.env),
            "velocity": self._walking_velocity_command() if velocity is None else velocity,
            "base_lin_vel": walking_mdp.base_lin_vel(self.env),
            "base_ang_vel": walking_mdp.base_ang_vel(self.env),
            "projected_gravity": walking_mdp.projected_gravity(self.env) if gravity is None else gravity,
        }

    def observation(self, key, *, _walking_state=None):
        """重建旧策略输入列；裁剪后缩放，不添加推理噪声或更新归一化。"""
        if key in ("walking", "omni"):
            self._update_memory()
        if key == "snake":
            return self._snake_observation()
        env = self.env
        cfg = self.configs[key]
        action = env.action_manager.get_term("joint_pos")
        command = env.command_manager.get_term("transform")
        if key == "transform":
            # 仅让纯函数读取临时代理，绝不运行旧缓存函数或 ObservationManager。
            mirror_action = SimpleNamespace(commanded_target=expand_mirror(project_mirror(action.commanded_target)), raw_action=project_mirror(action.raw_action * action.rate_cfg.max_target_speed / cfg.actions["joint_pos"].max_target_speed))
            proxy = SimpleNamespace(cfg=cfg, scene=env.scene, device=env.device, num_envs=env.num_envs, action_manager=SimpleNamespace(get_term=lambda name: mirror_action))
            pieces = []
            for term in teacher_reference_observation_groups()["actor"].terms.values():
                value = term.func(proxy, **term.params)
                if term.clip is not None:
                    value = value.clamp(*term.clip)
                if term.scale is not None:
                    value = value * term.scale
                pieces.append(value)
            pieces.append(torch.where(command.standing_mode, 1.0, -1.0).unsqueeze(-1))
            return torch.cat(pieces, dim=-1)
        state = self._walking_observation_state() if _walking_state is None else _walking_state
        data = env.scene["robot"].data
        action_cfg = cfg.actions["joint_pos"]
        scale = _joint_values(action_cfg.scale, data.joint_pos)
        offset = self.defaults[key] if action_cfg.use_default_offset else _joint_values(action_cfg.offset, data.joint_pos)
        last_position_action = (action.commanded_target - offset) / scale
        # 步态时钟与初始航向必须对学生可见；禁止教师另用隐藏条件。
        frequency = getattr(command, "gait_frequency", self._frequency[key])
        phase = env.episode_length_buf.float() * env.step_dt * frequency * (2 * torch.pi)
        current = state["current"]
        initial = getattr(command, "initial_heading", self._initial_forward[key])
        yaw_error = torch.atan2(current[:, 0] * initial[:, 1] - current[:, 1] * initial[:, 0], (current[:, :2] * initial[:, :2]).sum(-1))[:, None]
        velocity = state["velocity"]
        values = {
            "base_lin_vel": state["base_lin_vel"],
            "base_ang_vel": state["base_ang_vel"],
            "projected_gravity": state["projected_gravity"],
            "yaw_error": yaw_error,
            "joint_pos": data.joint_pos - self.defaults[key],
            "joint_vel": data.joint_vel,  # 原教师默认关节速度为零。
            "actions": last_position_action,
            "phase": torch.stack((phase.sin(), phase.cos()), -1),
            "frequency": ((frequency - cfg.step_frequency_min) / max(cfg.step_frequency_max - cfg.step_frequency_min, 1e-6))[:, None],
            "target_speed": velocity[:, :1],
            "velocity_command": velocity,
        }
        pieces = []
        for name, term in cfg.observations["actor"].terms.items():
            value = values[name]
            if term.clip is not None:
                value = value.clamp(*term.clip)
            if term.scale is not None:
                value = value * term.scale
            pieces.append(value)
        return torch.cat(pieces, -1)

    def _convert(self, key, output, env_ids=None):
        env = self.env
        action = env.action_manager.get_term("joint_pos")
        cfg = self.configs[key].actions["joint_pos"]
        target = action.commanded_target
        limits = action._target_limits()
        if env_ids is not None:
            target = target[env_ids]
            if limits.shape[0] != 1:
                limits = limits[env_ids]
        if key == "snake":
            teacher_target = target.clamp(-cfg.target_limit, cfg.target_limit)
            desired = (teacher_target + output.clamp(-1, 1) * cfg.max_target_speed * env.step_dt).clamp(-cfg.target_limit, cfg.target_limit)
        elif key == "transform":
            # 教师只看到镜像目标，必须从同一目标积分；完整投影同时消除学生的不对称漂移。
            mirror_target = project_mirror(target)
            desired = expand_mirror(mirror_target + output.clamp(-1, 1) * cfg.max_target_speed * env.step_dt)
            # 同时遵守教师自身的命名目标区间与学生可达区间。
            for i, name in enumerate(JOINT_NAMES):
                if name in cfg.joint_target_bounds:
                    desired[:, i].clamp_(*cfg.joint_target_bounds[name])
        else:
            if cfg.clip is not None:
                raise ValueError("位置教师 clip 配置发生变化，请更新动作适配器")
            scale = _joint_values(cfg.scale, target)
            offset = self.defaults[key] if cfg.use_default_offset else _joint_values(cfg.offset, target)
            desired = output * scale + offset
        return position_to_rate(desired, target, limits, env.step_dt, action.rate_cfg.max_target_speed)

    @torch.no_grad()
    def query(self) -> TeacherTargets:
        env = self.env
        if any(key in self.models for key in ("walking", "omni")):
            self._update_memory()
        command = env.command_manager.get_term("transform")
        action = env.action_manager.get_term("joint_pos")
        requested = command.requested_velocity
        quiet = requested.norm(dim=-1) < .01
        robot = env.scene["robot"]
        base = robot.find_bodies("base")[0][0]
        height = robot.data.body_link_pos_w[:, base, 2] - env.scene.env_origins[:, 2]
        velocity = gravity = None
        masks = {}
        if any(key in self.models for key in ("walking", "omni")):
            velocity = self._walking_velocity_command()
            moving = requested.abs().amax(-1) > .025
            translation = (velocity[:, :2].norm(dim=-1) > .025) & (requested[:, :2].norm(dim=-1) > .025)
            gravity = walking_mdp.projected_gravity(env)
            # Both legacy walking teachers share the same standing pose.
            standing_key = "walking" if "walking" in self.models else "omni"
            near_standing = (robot.data.joint_pos - self.defaults[standing_key]).abs().amax(-1) < 1.2
            target_near_standing = (action.commanded_target - self.defaults[standing_key]).abs().amax(-1) < 1.2
            upright = (gravity[:, 2].abs() < .55) & (height > .16) & near_standing & target_near_standing
            walk = command.standing_mode & ~command.transition_mask & moving & translation & upright
            walk &= (velocity[:, 2].abs() < .2) & (requested[:, 2].abs() < .2)
            forward = (velocity[:, 0] > .025) & (velocity[:, 1].abs() < .025) & (velocity[:, 2].abs() < .05)
            masks.update(walking=walk & forward, omni=walk & ~forward)
        if "transform" in self.models:
            mirror_error = (action.commanded_target - expand_mirror(project_mirror(action.commanded_target))).abs().amax(-1)
            joint_mirror_error = (robot.data.joint_pos - expand_mirror(project_mirror(robot.data.joint_pos))).abs().amax(-1)
            teacher_bounds_valid = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
            for i, name in enumerate(JOINT_NAMES):
                bounds = self.configs["transform"].actions["joint_pos"].joint_target_bounds.get(name)
                if bounds is not None:
                    teacher_bounds_valid &= (action.commanded_target[:, i] >= bounds[0] - .5) & (action.commanded_target[:, i] <= bounds[1] + .5)
            transform = (command.transition_mask | quiet) & (mirror_error < 0.75) & (joint_mirror_error < 0.75) & teacher_bounds_valid
            masks["transform"] = transform
        straighten = self._straightening_mask(height) if "transform" in self.models else torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        masks = {key: masks[key] & ~straighten for key in self.models if key in masks}
        snake_observation = None
        if "snake" in self.models:
            # 归档只验证世界+X低位前进；其他指令与变形继续由PPO学习。
            snake_observation = self._snake_observation()
            proxy = self._snake_proxy
            cfg = self.configs["snake"]
            heading = getattr(command, "heading", None)
            if heading is None:
                world_velocity = command.velocity_command
                world_requested = requested
            else:
                left = torch.stack((-heading[:, 1], heading[:, 0]), -1)
                world_velocity = torch.cat((heading * command.velocity_command[:, :1] + left * command.velocity_command[:, 1:2], command.velocity_command[:, 2:]), -1)
                world_requested = torch.cat((heading * requested[:, :1] + left * requested[:, 1:2], requested[:, 2:]), -1)
            def covered(value):
                return (value[:, 0] > .025) & (value[:, 1:].abs().amax(-1) < .01)
            native_limit = cfg.actions["joint_pos"].target_limit
            snake = (~command.standing_mode & ~command.transition_mask & ~straighten
                     & covered(world_requested) & covered(world_velocity)
                     & (snake_mdp.head_forward(proxy)[:, 0] > math.cos(cfg.head_yaw_limit))
                     & (height < .12) & (snake_mdp.head_height(proxy) < .12)
                     & (snake_mdp.head_tilt(proxy) < cfg.head_tilt_limit)
                     & (snake_mdp.head_yaw_abs_deviation(proxy) < cfg.head_yaw_limit)
                     & (snake_mdp.body_yaw_abs_deviation(proxy) < cfg.body_yaw_limit)
                     # 目标限位不等于物理位置硬限位；保留伺服跟踪超调余量。
                     & (robot.data.joint_pos.abs().amax(-1) <= native_limit + .1)
                     & (action.commanded_target.abs().amax(-1) <= native_limit))
            # 世界+X请求不能掩盖学生自己的后退／侧移／转向意图。
            snake &= covered(requested) & covered(command.velocity_command)
            masks["snake"] = snake
        result = action.commanded_target.new_zeros(env.num_envs, 6)
        weights = result.new_zeros(env.num_envs)
        teacher_ids = torch.full((env.num_envs,), -1, device=self.device, dtype=torch.long)
        counts = {}
        walking_state = None
        for teacher_id, key in enumerate(self.teacher_names):
            mask = masks[key]
            env_ids = mask.nonzero(as_tuple=False).flatten()
            counts[key] = env_ids.numel()
            if not counts[key]:
                continue
            if key in ("walking", "omni") and walking_state is None:
                walking_state = self._walking_observation_state(velocity, gravity)
            # 蛇形全量观测已用于覆盖范围判定；仅在本次查询内复用，避免历史缓存陈旧。
            observation = snake_observation if key == "snake" else self.observation(key, _walking_state=walking_state)
            assert observation is not None
            observation = observation[env_ids]
            inputs = TensorDict({"actor": observation}, batch_size=[env_ids.numel()])
            output = self.models[key](inputs, stochastic_output=False)
            converted = self._convert(key, output, env_ids=env_ids)
            finite = torch.isfinite(observation).all(-1) & torch.isfinite(output).all(-1) & torch.isfinite(converted).all(-1)
            selected = env_ids[finite]
            result[selected] = converted[finite]
            weights[selected] = 1.0
            teacher_ids[selected] = teacher_id
            counts[key] = selected.numel()
        if straighten.any():
            # 仅提供蒸馏标签；是否执行教师动作仍由现有学生采样机制决定。
            rates = position_to_rate(torch.zeros_like(action.commanded_target), action.commanded_target, action._target_limits(), env.step_dt, action.rate_cfg.max_target_speed)
            result[straighten] = rates[straighten]
            weights[straighten] = 1.0
            teacher_ids[straighten] = self.teacher_names.index("transform")
            counts["transform"] += int(straighten.sum())
        return TeacherTargets(result, weights, counts, teacher_ids)
