"""独立固定场景的学生能力评测，不读取教师或改变训练环境。"""

import math
import random
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from zbot_rl_mjlab.experiments.universal import universal_env_cfg

from .robot import DEFAULT_JOINT_POS
from .tasks.transformation.poses import SNAKE_ROOT_POS, SNAKE_ROOT_QUAT
from .tasks.transformation.roll_reset import (
    collision_radial_envelope,
    rolled_root_states,
)
from .tasks.universal import mdp
from .tasks.universal.commands import UniversalCommandCfg

CAPABILITIES = ('forward', 'lateral', 'descent', 'ascent')
MOTION_DIRECTIONS = {'forward': (1., 0.), 'backward': (-1., 0.), 'left': (0., 1.), 'right': (0., -1.)}
# 独立于训练随机姿态库：每种姿态分别覆盖四个固定侧滚角。
ASCENT_POSES = (
    (0.,) * 6, (.15, 0., 0., 0., 0., 0.), (0., 0., .15, 0., 0., 0.),
    (0., 0., 0., 0., -.15, 0.), (.15,) * 6, (.2, 0., -.2, 0., .2, 0.),
    (0., .2, 0., -.2, 0., .2), (.15, -.1, .15, .1, -.15, .1),
)


def _evaluation_envelope(env, pose, reference):
    """环境内缓存固定姿态包络，避免八种姿态挤出原生函数的小型缓存。"""
    if not hasattr(env, '_evaluation_pose_envelopes'):
        env._evaluation_pose_envelopes = {}
    cache = env._evaluation_pose_envelopes
    if pose not in cache:
        radial, low, high = collision_radial_envelope(pose)
        cache[pose] = (reference.new_tensor(radial), low, high)
    return cache[pose]


def _validate_forward_settings(speed, tolerance, direction='forward'):
    """外部可以逐次调整前进档位，构造及每次验收都检查参数。"""
    if not isinstance(direction, str) or direction not in MOTION_DIRECTIONS:
        raise ValueError('运动方向必须为 forward、backward、left 或 right')
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('前进验收速度必须为有限正数')
    if not math.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError('前进验收相对容差必须为有限数且位于 [0, 1)')


def evaluate_capabilities(velocity, ready, finite, success_out=None, *, forward_speed=.08, forward_tolerance=.5, strict_forward=False, motion_direction='forward', include_snake=False):
    """末半段至少九成时间构型到位，且每类至少四分之三环境通过。"""
    _validate_forward_settings(forward_speed, forward_tolerance, motion_direction)
    capabilities = CAPABILITIES + ('snake',) if include_snake else CAPABILITIES
    count = len(capabilities)
    if velocity.shape[1] < count or velocity.shape[1] % count:
        raise ValueError('能力评测环境数量必须能均分所有能力组')
    groups = torch.arange(velocity.shape[1], device=velocity.device) // (velocity.shape[1] // count)
    if motion_direction != 'forward':
        # 仅把第零组投影到所请求的运动轴；其余三组保持原验收坐标。
        dx, dy = MOTION_DIRECTIONS[motion_direction]
        original = velocity
        velocity = velocity.clone()
        selected = groups == 0
        velocity[:, selected, 0] = original[:, selected, 0] * dx + original[:, selected, 1] * dy
        velocity[:, selected, 1] = -original[:, selected, 0] * dy + original[:, selected, 1] * dx
    mean = velocity.mean(0)
    absolute = velocity.abs().mean(0)
    finite = finite & torch.isfinite(velocity).all(dim=(0, 2))
    success = (ready.float().mean(0) >= .9) & finite
    # 平移至少达到目标一半，并限制侧向滑移和无意转向。
    forward = (mean[:, 0] >= forward_speed * (1 - forward_tolerance)) & (mean[:, 0] <= forward_speed * (1 + forward_tolerance)) & (mean[:, 1].abs() <= .04) & (absolute[:, 1] <= .10)
    lateral = (mean[:, 1] >= .025) & (mean[:, 1] <= .075) & (mean[:, 0].abs() <= .04) & (absolute[:, 0] <= .10)
    # 同时要求持续正确方向推进，不能用一次冲刺补足其余静止时间。
    forward_lower = forward_speed * (.5 if strict_forward else .25)
    forward_motion = (velocity[..., 0] >= forward_lower) & (velocity[..., 0] <= forward_speed * 2)
    forward &= forward_motion.float().mean(0) >= .6
    lateral &= ((velocity[..., 1] >= .0125) & (velocity[..., 1] <= .10)).float().mean(0) >= .6
    snake_motion = (velocity[..., 0] >= .04) & (velocity[..., 0] <= .16)
    snake = (mean[:, 0] >= .04) & (mean[:, 0] <= .12) & (mean[:, 1].abs() <= .04) & (absolute[:, 1] <= .10)
    snake &= snake_motion.float().mean(0) >= .6
    success &= torch.where(groups == 0, forward, torch.where(groups == 1, lateral, torch.where(groups == 4, snake, True)))
    # 基座会随步态摆动：同时约束净转向与摆动总量，避免正负转速抵消。
    success &= ((groups >= 2) & (groups != 4)) | ((mean[:, 2].abs() <= .15) & (absolute[:, 2] <= .8))
    result = {f'{name}_success': success[groups == i].float().mean().item() for i, name in enumerate(capabilities)}
    result['forward_velocity'] = mean[groups == 0, 0].mean().item()
    for i, name in enumerate(capabilities):
        result[f'{name}_ready_fraction'] = ready[:, groups == i].float().mean().item()
    result['forward_motion_fraction'] = forward_motion[:, groups == 0].float().mean().item()
    result['lateral_motion_fraction'] = ((velocity[:, groups == 1, 1] >= .0125) & (velocity[:, groups == 1, 1] <= .10)).float().mean().item()
    result['forward_cross_speed'] = absolute[groups == 0, 1].mean().item()
    result['forward_yaw_speed'] = absolute[groups == 0, 2].mean().item()
    result['lateral_cross_speed'] = absolute[groups == 1, 0].mean().item()
    result['lateral_yaw_speed'] = absolute[groups == 1, 2].mean().item()
    result['lateral_velocity'] = mean[groups == 1, 1].mean().item()
    result['validation_passed'] = float(all(result[f'{name}_success'] >= .75 for name in capabilities))
    if include_snake:
        result['snake_velocity'] = mean[groups == 4, 0].mean().item()
        result['snake_motion_fraction'] = snake_motion[:, groups == 4].float().mean().item()
        result['snake_cross_speed'] = absolute[groups == 4, 1].mean().item()
        result['snake_yaw_speed'] = absolute[groups == 4, 2].mean().item()
    if success_out is not None:
        success_out.copy_(success)
    return result


def reset_evaluation_pose(env, env_ids):
    """固定能力初态；部署起立覆盖八种姿态，蛇爬使用归档教师朝前初态。"""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene['robot']
    root = robot.data.default_root_state[ids].clone()
    root[:, 7:] = 0
    q = root.new_tensor(DEFAULT_JOINT_POS).expand(len(ids), -1).clone()
    _, standing_low, _ = _evaluation_envelope(env, DEFAULT_JOINT_POS, root)
    root[:, :3] = env.scene.env_origins[ids] + root.new_tensor([0., -.06, .003 - standing_low])
    root[:, 3:7] = root.new_tensor([1., 0., 0., 0.])
    include_snake = getattr(env.cfg, 'snake_teacher_enabled', False)
    width = env.num_envs // (5 if include_snake else 4)
    snake = ids // width == 3
    if snake.any():
        source = root[snake].clone()
        source[:, :3] = root.new_tensor(SNAKE_ROOT_POS)
        source[:, 3:7] = root.new_tensor(SNAKE_ROOT_QUAT)
        local = ids[snake] % width
        rolls = root.new_tensor([-20., 0., 20., 40.])[local % 4] * math.pi / 180
        bent = getattr(env.cfg, 'straighten_before_stand', False) and width >= 32
        poses = ASCENT_POSES if bent else ((0.,) * 6,)
        pose_ids = (local % 32) // 4 if bent else torch.zeros_like(local)
        snake_root, snake_q = source.clone(), q[snake].clone()
        for index, pose in enumerate(poses):
            selected = pose_ids == index
            if not selected.any():
                continue
            radial, zmin, zmax = _evaluation_envelope(env, pose, root)
            snake_root[selected] = rolled_root_states(source[selected], env.scene.env_origins[ids[snake][selected]], rolls[selected], radial, zmin, zmax, .0001)
            snake_q[selected] = root.new_tensor(pose)
        root[snake], q[snake] = snake_root, snake_q
    if include_snake:
        crawling = ids // width == 4
        root[crawling, :3] = env.scene.env_origins[ids[crawling]] + root.new_tensor([0., 0., .0555])
        root[crawling, 3:7] = root.new_tensor([math.sqrt(.5), 0., -math.sqrt(.5), 0.])
        q[crawling] = 0.
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    robot.write_root_state_to_sim(root, env_ids=ids)


def evaluation_env_cfg(env_cfg: ManagerBasedRlEnvCfg | None = None, *, num_envs=16, randomized=True):
    """深复制训练配置并保留传感器历史结构；标称评测仅移除物理随机化及外扰事件。"""
    cfg = deepcopy(env_cfg) if env_cfg is not None else universal_env_cfg(play=True)
    if getattr(cfg, 'snake_teacher_enabled', False):
        num_envs = max(160 if getattr(cfg, 'straighten_before_stand', False) else 20, math.ceil(num_envs / 5) * 5)
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    cfg.seed = None
    command = cast(UniversalCommandCfg, cfg.commands['transform'])
    command.training = False
    command.mode = 'stand'
    command.auto_switch_on_success = False
    for group in cfg.observations.values():
        group.enable_corruption = False
    if not randomized:
        cfg.events = {name: event for name, event in cfg.events.items()
                      if not name.startswith(('randomize_', 'push_')) and name != 'base_disturbance'}
    elif 'base_disturbance' in cfg.events:
        # 鲁棒验收不能让未受扰的大多数环境掩盖受扰环境全部失败。
        cfg.events['base_disturbance'].params['disturbed_fraction'] = 1.
    cfg.events['reset_transition'] = EventTermCfg(func=reset_evaluation_pose, mode='reset')
    return cfg


class UniversalEvaluator:
    """惰性创建小型独立环境，保留网络训练模式及训练随机数状态。"""

    def __init__(self, device='cpu', num_envs=16, seconds=10., env_cfg=None, randomized=True, forward_speed=.08, forward_tolerance=.5, strict_forward=False, motion_direction='forward', keyboard_protocol=False):
        self.include_snake = getattr(env_cfg, 'snake_teacher_enabled', False)
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 16 or (not self.include_snake and num_envs % 4) or seconds < 2:
            raise ValueError('能力评测至少需要 16 个环境、4 的倍数和 2 秒时长')
        _validate_forward_settings(forward_speed, forward_tolerance, motion_direction)
        self.motion_direction = motion_direction
        self.forward_speed = forward_speed
        self.forward_tolerance = forward_tolerance
        self.strict_forward = strict_forward
        self.keyboard_protocol = keyboard_protocol
        self.device = torch.device(device)
        self.num_envs = max(128, num_envs) if getattr(env_cfg, 'straighten_before_stand', False) else num_envs
        if self.include_snake:
            self.num_envs = max(160 if getattr(env_cfg, 'straighten_before_stand', False) else 20, math.ceil(num_envs / 5) * 5)
        self.seconds = seconds
        self.env_cfg = deepcopy(env_cfg)
        self.randomized = randomized
        self.rl_iteration = 0
        self.env = None
        self.wrapper = None
        self.foundation_samples = None

    def _build(self):
        cfg = evaluation_env_cfg(self.env_cfg, num_envs=self.num_envs, randomized=self.randomized)
        self.env = ManagerBasedRlEnv(cfg, device=str(self.device))
        self.wrapper = RslRlVecEnvWrapper(self.env)

    def __call__(self, actor):
        _validate_forward_settings(self.forward_speed, self.forward_tolerance, self.motion_direction)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        training = actor.training
        self.foundation_samples = None
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        try:
            actor.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                random.seed(42)
                np.random.seed(42)
                torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
                if self.device.type == 'cuda':
                    state = torch.Generator(device=self.device).manual_seed(42).get_state()
                    torch.cuda.set_rng_state(state, self.device)
                if self.env is None:
                    self._build()
                env, wrapper = self.env, self.wrapper
                assert env is not None and wrapper is not None
                physical = cast(Any, env)
                physical._student_rl_iteration = self.rl_iteration
                physical._student_pretraining = False
                wrapper.reset()
                command = mdp.command_term(env)
                groups = torch.arange(self.num_envs, device=self.device) // (self.num_envs // (5 if self.include_snake else 4))
                dx, dy = MOTION_DIRECTIONS[self.motion_direction]
                targets = ('stand', 'stand', 'snake', 'stand', 'snake') if self.include_snake else ('stand', 'stand', 'snake', 'stand')
                for i, target in enumerate(targets):
                    ids = (groups == i).nonzero().flatten()
                    command.set_target(target, ids)
                    vx = self.forward_speed * dx if i == 0 else (.08 if i == 4 else 0.)
                    vy = self.forward_speed * dy if i == 0 else (.05 if i == 1 else 0.)
                    if self.keyboard_protocol and i == 0:
                        vx, vy = (.08 if self.motion_direction != 'forward' else 0.), 0.
                    command.set_velocity(vx, vy, 0., ids)
                # reset() 已缓存旧指令；仅在本次独立场景初态清空并回填全部历史帧。
                env.observation_manager.reset()
                env.observation_manager.compute(update_history=True, env_ids=torch.arange(self.num_envs, device=self.device))
                obs = wrapper.get_observations()
                velocities, ready = [], []
                trajectory_observations, trajectory_actions = [], []
                finite = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                no_fall = torch.ones_like(finite)
                evaluation_steps = math.ceil(self.seconds / env.step_dt)
                preparation = 3.7 if self.keyboard_protocol else 0.
                steps = math.ceil((self.seconds + preparation) / env.step_dt)
                measurement_start = steps - evaluation_steps + evaluation_steps // 2
                target_steps = [math.ceil((2.2 + phase * .5) / env.step_dt - 1e-9) for phase in range(4)]
                stop_step = math.ceil(2. / env.step_dt - 1e-9)
                local_phase = torch.arange(self.num_envs, device=self.device) % 4
                for step in range(steps):
                    if self.keyboard_protocol:
                        if self.motion_direction != 'forward' and step == stop_step:
                            command.set_velocity(0., 0., 0., (groups == 0).nonzero().flatten())
                        for phase, target_step in enumerate(target_steps):
                            if step == target_step:
                                ids = ((groups == 0) & (local_phase == phase)).nonzero().flatten()
                                # 模拟键盘改向，只更新请求；保留改向前的真实传感器历史。
                                command.set_velocity(self.forward_speed * dx, self.forward_speed * dy, 0., ids)
                    actions = actor(obs, stochastic_output=False)
                    # 保存真实闭环的起步、过渡和稳态；不得用教师轨迹代替学生验收轨迹。
                    with torch.inference_mode(False), torch.no_grad():
                        trajectory_observations.append(obs.detach().clone())
                        trajectory_actions.append(actions.detach().clone())
                    obs, rewards, _, _ = wrapper.step(actions)
                    finite &= torch.isfinite(obs['actor']).all(-1) & torch.isfinite(rewards) & ~mdp.unsafe(env)
                    if self.strict_forward:
                        # 手动指令可能屏蔽终止；前进验收独立检查整个闭环是否曾跌倒。
                        height = mdp.base_height(env)
                        no_fall &= torch.isfinite(height) & (height >= .10)
                    if step >= measurement_start:
                        velocity = mdp.tracking_velocity(env).clone()
                        if self.include_snake:
                            # 蛇教师初态朝世界 +X；还原世界质心速度，防止局部航向投影掩盖侧滑。
                            heading = mdp.heading(env)
                            crawling = groups == 4
                            vx, vy = velocity[crawling, 0].clone(), velocity[crawling, 1].clone()
                            velocity[crawling, 0] = vx * heading[crawling, 0] - vy * heading[crawling, 1]
                            velocity[crawling, 1] = vx * heading[crawling, 1] + vy * heading[crawling, 0]
                        velocities.append(velocity)
                        ready.append(mdp.configuration_ready(env).clone())
                successful = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                if self.strict_forward:
                    finite &= (groups != 0) | no_fall
                result = evaluate_capabilities(torch.stack(velocities), torch.stack(ready), finite, successful, forward_speed=self.forward_speed, forward_tolerance=self.forward_tolerance, strict_forward=self.strict_forward, motion_direction=self.motion_direction, include_snake=self.include_snake)
                if self.keyboard_protocol:
                    result['keyboard_protocol'] = 1.
                if self.strict_forward:
                    result['forward_no_fall_fraction'] = no_fall[groups == 0].float().mean().item()
                if getattr(self.env_cfg, 'straighten_before_stand', False):
                    ascent = successful[groups == 3]
                    pose_ids = (torch.arange(len(ascent), device=self.device) % 32) // 4
                    for index in range(len(ASCENT_POSES)):
                        result[f'ascent_pose_{index}_success'] = ascent[pose_ids == index].float().mean().item()
                    result['ascent_bent_success'] = ascent[pose_ids != 0].float().mean().item()
                    result['validation_passed'] *= float(result['ascent_success'] >= .9 and all(result[f'ascent_pose_{index}_success'] >= .75 for index in range(len(ASCENT_POSES))))
                if successful.any():
                    with torch.inference_mode(False), torch.no_grad():
                        selected = successful.repeat(steps).clone()
                        observations = cast(TensorDict, torch.cat(trajectory_observations, dim=0))[selected].clone()
                        actions = torch.cat(trajectory_actions, dim=0)[selected].clone()
                        teacher_ids = torch.where(groups == 4, 3, groups.clamp(max=2)).repeat(steps)[selected].clone()
                        held_out = torch.arange(len(actions), device=self.device) % 5 == 0
                        self.foundation_samples = (observations, actions, actions.new_ones(len(actions)), teacher_ids, held_out)
                return result
        finally:
            actor.train(training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = self.wrapper = None
        self.foundation_samples = None
