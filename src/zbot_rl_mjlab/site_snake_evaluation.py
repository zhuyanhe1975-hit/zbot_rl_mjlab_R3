"""Independent two-speed, four-heading qualification for site-frame snake motion."""

import math
import random
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul

from .tasks.snake import mdp as snake
from .tasks.snake import site_omni as site_snake
from .tasks.transformation.roll_reset import collision_radial_envelope
from .tasks.universal import mdp
from .tasks.universal.omni import reset_omni

NAMES = ('stop', 'forward', 'backward', 'left', 'right', 'yaw_left', 'yaw_right', 'mixed_positive', 'mixed_negative')
SPEEDS = ((.08, .2), (.15, .5))
CASES = tuple((name, speed, yaw) for name in NAMES for speed in range(2) for yaw in (0, 90, 180, 270))


def targets(reference):
    result = []
    for name, level, _ in CASES:
        speed, yaw = SPEEDS[level]
        result.append({'stop': (0., 0., 0.), 'forward': (speed, 0., 0.),
                       'backward': (-speed, 0., 0.), 'left': (0., speed, 0.),
                       'right': (0., -speed, 0.), 'yaw_left': (0., 0., yaw),
                       'yaw_right': (0., 0., -yaw), 'mixed_positive': (speed, speed, yaw),
                       'mixed_negative': (-speed, -speed, -yaw)}[name])
    return reference.new_tensor(result)


def score_site_snake(velocity, failed):
    """Score every speed and heading separately; failures cover the entire trial."""
    if velocity.ndim != 3 or velocity.shape[-1] != 3 or not velocity.shape[0]:
        raise ValueError('velocity must be nonempty [time, environments, 3]')
    n = velocity.shape[1]
    if not n or n % len(CASES) or failed.shape != (n,):
        raise ValueError('require equally replicated 72 speed/heading cases and per-environment failures')
    replicas = n // len(CASES)
    target = targets(velocity).repeat_interleave(replicas, 0)
    tolerance = torch.maximum(target.abs() * .4, velocity.new_tensor((.02, .02, .12)))
    error = (velocity - target).abs()
    sustained = (error <= tolerance).all(-1).float().mean(0)
    mean = velocity.mean(0)
    signed_response = mean * target.sign()
    response = ((target == 0) | (signed_response >= target.abs() * .5)).all(-1)
    passed = (~failed.bool() & torch.isfinite(velocity).all(dim=(0, 2)) & response
              & (sustained >= .6) & (error.mean(0) <= tolerance).all(-1)
              & ((mean - target).abs() <= tolerance).all(-1))
    details = []
    metrics = {}
    scenario_passes = []
    for i, (name, level, yaw) in enumerate(CASES):
        selection = slice(i * replicas, (i + 1) * replicas)
        rate = passed[selection].float().mean().item()
        details.append({'scenario': name, 'speed': SPEEDS[level][0], 'initial_yaw_deg': yaw,
                        'target': target[i * replicas].tolist(), 'success_rate': rate,
                        'passed': rate >= .75,
                        'actual_mean': torch.nan_to_num(mean[selection]).mean(0).tolist(),
                        'failure_rate': failed[selection].float().mean().item(),
                        'sustained_fraction': sustained[selection].mean().item()})
    for name in NAMES:
        rows = [row for row in details if row['scenario'] == name]
        metrics[f'snake_{name}_success'] = sum(float(row['success_rate']) for row in rows) / 8
        scenario_passes.append(all(row['passed'] for row in rows))
    metrics['snake_omni_scenarios_passed'] = float(sum(scenario_passes))
    metrics['snake_omni_passed'] = float(all(scenario_passes))
    return metrics, {'passed': all(scenario_passes), 'cases': details,
                     'criteria': {'sustained_fraction': .6, 'per_case_success_fraction': .75,
                                  'minimum_signed_response_fraction': .5}}


def reset_site_snake_evaluation(env, env_ids):
    """Canonical straight snake with collision-clear height at four initial yaws."""
    reset_omni(env, env_ids)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    # Reset events run before command_manager.reset: keep its pose cache truthful.
    for name in ('_omni_initial_standing', '_omni_play_standing'):
        if hasattr(env, name):
            getattr(env, name)[ids] = False
    robot = env.scene['robot']
    root = robot.data.default_root_state[ids].clone()
    replicas = env.num_envs // len(CASES)
    yaw = (ids // replicas).remainder(4) * (math.pi / 2)
    rotation = torch.zeros_like(root[:, 3:7])
    rotation[:, 0], rotation[:, 3] = (yaw / 2).cos(), (yaw / 2).sin()
    root[:, 3:7] = quat_mul(rotation, root.new_tensor([2**-.5, 0., -2**-.5, 0.]).expand(len(ids), -1))
    root[:, :3] = env.scene.env_origins[ids]
    up = torch.zeros_like(root[:, :3])
    up[:, 2] = 1.
    local_up = quat_apply_inverse(root[:, 3:7], up)
    radial, low, high = collision_radial_envelope((0.,) * 6)
    zmin = (local_up[:, :2] @ root.new_tensor(radial).T).amin(-1)
    zmin += torch.minimum(local_up[:, 2] * low, local_up[:, 2] * high)
    root[:, 2] += .003 - zmin
    root[:, 7:] = 0.
    joints = robot.data.default_joint_pos[ids].clone().zero_()
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
    robot.write_root_state_to_sim(root, env_ids=ids)
    term = cast(Any, mdp.command_term(env))
    term.heading_initialized[ids] = False
    term.initialize_task_state(ids, torch.zeros(len(ids), dtype=torch.bool, device=env.device))


class SiteSnakeEvaluator:
    """No automatic resets, task switches, or cross-case averaging."""

    def __init__(self, device='cpu', num_envs=72, seconds=10., env_cfg=None, randomized=False):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError('num_envs must be a positive integer')
        if not math.isfinite(seconds) or seconds < 2:
            raise ValueError('seconds must be finite and at least two')
        if env_cfg is None or not getattr(env_cfg, 'site_snake_omni', False):
            raise ValueError('requires the independent site snake experiment configuration')
        self.device = torch.device(device)
        self.num_envs = max(72, math.ceil(num_envs / 72) * 72)
        self.seconds, self.randomized = seconds, randomized
        self.env_cfg = deepcopy(env_cfg)
        self.env = self.wrapper = None
        self.last_report = {}
        self.rl_iteration = 0
        self.foundation_samples = None

    def _build(self):
        cfg = deepcopy(self.env_cfg)
        cfg.scene.num_envs, cfg.auto_reset, cfg.seed = self.num_envs, False, None
        cfg.terminations.clear()
        # Rewards are not acceptance criteria; avoid termination-dependent reward callbacks.
        cfg.rewards.clear()
        cfg.metrics.clear()
        command = cast(Any, cfg.commands['transform'])
        command.training, command.mode = False, 'snake'
        command.auto_switch_on_success = False
        command.play_random_velocity = False
        command.randomize_play_modes = False
        required_limits = (max(speed for speed, _ in SPEEDS),) * 2 + (max(yaw for _, yaw in SPEEDS),)
        if any(limit < required for limit, required in zip(command.velocity_limits, required_limits, strict=True)):
            raise ValueError('qualification command limits must cover both speeds without clamping')
        for group in cfg.observations.values():
            group.enable_corruption = False
        if not self.randomized:
            cfg.events = {name: event for name, event in cfg.events.items()
                          if not name.startswith(('randomize_', 'push_')) and name != 'base_disturbance'}
        cfg.events['reset_transition'] = EventTermCfg(func=reset_site_snake_evaluation, mode='reset')
        self.env = ManagerBasedRlEnv(cfg, device=str(self.device))
        self.wrapper = RslRlVecEnvWrapper(self.env)

    def _commands(self, term, initial):
        requested = targets(torch.empty(0, device=self.device)).repeat_interleave(self.num_envs // 72, 0)
        if initial:
            requested.zero_()
            for index, (name, level, _) in enumerate(CASES):
                if name == 'stop':
                    start = index * (self.num_envs // 72)
                    requested[start:start + self.num_envs // 72, 0] = SPEEDS[level][0]
        term.set_target('snake', torch.arange(self.num_envs, device=self.device))
        width = self.num_envs // 72
        for index in range(len(CASES)):
            ids = torch.arange(index * width, (index + 1) * width, device=self.device)
            term.set_velocity(*requested[index * width].tolist(), ids)

    def __call__(self, actor):
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == 'cuda' else [])
        training, python_rng, numpy_rng = actor.training, random.getstate(), np.random.get_state()
        try:
            actor.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                random.seed(42)
                np.random.seed(42)
                torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
                if self.device.type == 'cuda':
                    torch.cuda.set_rng_state(torch.Generator(device=self.device).manual_seed(42).get_state(), self.device)
                if self.env is None:
                    self._build()
                env, wrapper = self.env, self.wrapper
                assert env is not None and wrapper is not None
                cast(Any, env)._student_rl_iteration = self.rl_iteration
                wrapper.reset()
                term = mdp.command_term(env)
                self._commands(term, True)
                env.observation_manager.reset()
                env.observation_manager.compute(update_history=True, env_ids=torch.arange(self.num_envs, device=self.device))
                obs = wrapper.get_observations()
                failed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                switch = math.ceil(1. / env.step_dt)
                steps = switch + math.ceil(self.seconds / env.step_dt)
                measurement = switch + (steps - switch) // 2
                velocities = []
                for step in range(steps):
                    if step == switch:
                        self._commands(term, False)
                    actions = actor(obs, stochastic_output=False)
                    obs, _, _, _ = wrapper.step(actions)
                    data = env.scene['robot'].data
                    position = data.root_link_pos_w - env.scene.env_origins
                    failed |= ~torch.isfinite(obs['actor']).all(-1) | ~torch.isfinite(data.joint_pos).all(-1)
                    failed |= (~torch.isfinite(position).all(-1) | (position[:, :2].norm(dim=-1) > 4.)
                               | (position[:, 2] < -.1) | (position[:, 2] > 1.))
                    failed |= ~torch.isfinite(actions).all(-1)
                    tilt, force = site_snake.site_tilt(env), mdp.normalized_force(env)
                    failed |= (~torch.isfinite(tilt) | ~torch.isfinite(force)
                               | (tilt > cast(Any, env.cfg).site_tilt_limit + 1e-6))
                    failed |= snake.self_collision(env) | (force > 8.)
                    velocity = mdp.tracking_velocity(env)
                    failed |= ~torch.isfinite(velocity).all(-1)
                    if step >= measurement:
                        velocities.append(velocity.clone())
                metrics, self.last_report = score_site_snake(torch.stack(velocities), failed)
                base_reference = getattr(self.env_cfg, 'motion_site', 'foot_0_sole') == 'base_motion'
                protocol = ('site_snake_omni_v3' if getattr(self.env_cfg, 'archive_site_checkpoint', False)
                            else 'site_snake_base_omni_v5' if base_reference else 'site_snake_omni_v4')
                self.last_report.update(protocol=protocol, seconds=self.seconds, duration_eligible=self.seconds >= 10.,
                                        initial_yaws_deg=[0, 90, 180, 270], speeds=list(SPEEDS),
                                        randomized=self.randomized, frame='gravity_aligned_base_minus_z' if base_reference else 'gravity_aligned_foot_0_sole_minus_z',
                                        tracking_point='base' if base_reference else 'foot_0_sole')
                if self.seconds < 10.:
                    self.last_report['passed'] = False
                    metrics['snake_omni_passed'] = 0.
                return metrics
        finally:
            actor.train(training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = self.wrapper = None
