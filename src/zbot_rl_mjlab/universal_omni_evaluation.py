"""Independent signed, site-frame locomotion and keyboard transition acceptance."""

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

from .robot import DEFAULT_JOINT_POS
from .tasks.transformation.roll_reset import collision_radial_envelope
from .tasks.universal import mdp
from .universal_evaluation import evaluation_env_cfg

MOTIONS = (
    ('stop', (0., 0., 0.)),
    ('forward', (.08, 0., 0.)), ('backward', (-.08, 0., 0.)),
    ('left', (0., .05, 0.)), ('right', (0., -.05, 0.)),
    ('yaw_left', (0., 0., .3)), ('yaw_right', (0., 0., -.3)),
    ('mixed_positive', (.08, .05, .3)),
    ('mixed_negative', (-.08, -.05, -.3)),
)
SCENARIOS = tuple((f'{mode}_{name}', mode, command) for mode in ('biped', 'snake') for name, command in MOTIONS) + (
    ('descent', 'snake', (0., 0., 0.)), ('ascent', 'biped', (0., 0., 0.)),
)


def evaluate_omni_capabilities(velocity, ready, finite, *, no_fall=None):
    """Score sustained final-half traces in the *current* gravity-aligned site frame.

    Each scenario has equal-width replicas ordered by initial yaw (0, 90, 180,
    270 degrees). Every heading must pass; yaw commands have no heading lock.
    """
    count = len(SCENARIOS)
    if velocity.ndim != 3 or velocity.shape[-1] != 3 or not velocity.shape[0]:
        raise ValueError('velocity must have shape [time, environments, 3]')
    n = velocity.shape[1]
    if n < count * 4 or n % (count * 4):
        raise ValueError('each scenario requires equal replicas at all four initial yaws')
    if ready.shape != velocity.shape[:2] or finite.shape != (n,):
        raise ValueError('ready/finite shape does not match velocity')
    width = n // count
    targets = velocity.new_tensor([item[2] for item in SCENARIOS]).repeat_interleave(width, 0)
    # Active axes track within 50%; inactive axes must remain close to zero.
    tolerance = torch.where(targets != 0, targets.abs() * .5, targets.new_tensor([.04, .04, .15]))
    within = (velocity - targets).abs() <= tolerance
    sustained = within.all(-1).float().mean(0)
    mean_error = (velocity.mean(0) - targets).abs()
    absolute_error = (velocity - targets).abs().mean(0)
    successful = finite & torch.isfinite(velocity).all(dim=(0, 2))
    successful &= (ready.float().mean(0) >= .9) & (sustained >= .6)
    successful &= (mean_error <= tolerance).all(-1) & (absolute_error <= tolerance).all(-1)
    if no_fall is not None:
        if no_fall.shape != (n,):
            raise ValueError('no_fall shape does not match velocity')
        successful &= no_fall
    result = {}
    passed = True
    for i, (name, _, _) in enumerate(SCENARIOS):
        selection = slice(i * width, (i + 1) * width)
        fractions = successful[selection].reshape(-1, 4).float().mean(0)
        result[f'{name}_success'] = successful[selection].float().mean().item()
        result[f'{name}_ready_fraction'] = ready[:, selection].float().mean().item()
        result[f'{name}_motion_fraction'] = sustained[selection].mean().item()
        for yaw in range(4):
            result[f'{name}_yaw_{yaw * 90}_success'] = fractions[yaw].item()
        passed &= bool((fractions >= .75).all().item())
    result['validation_passed'] = float(passed)
    result['keyboard_protocol'] = 1.
    return result


def reset_omni_evaluation_pose(env, env_ids):
    """Canonical standing / horizontal snake, independently yawed and collision clear."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene['robot']
    root = robot.data.default_root_state[ids].clone()
    root[:, 7:] = 0
    groups = ids // (env.num_envs // len(SCENARIOS))
    # Descent begins standing; ascent begins snake.
    snake = ((groups >= len(MOTIONS)) & (groups < 2 * len(MOTIONS))) | (groups == len(SCENARIOS) - 1)
    q = root.new_tensor(DEFAULT_JOINT_POS).expand(len(ids), -1).clone()
    q[snake] = 0.
    root[:, 3:7] = root.new_tensor([1., 0., 0., 0.])
    root[snake, 3:7] = root.new_tensor([math.sqrt(.5), 0., -math.sqrt(.5), 0.])
    yaw = (ids % (env.num_envs // len(SCENARIOS))) % 4 * (math.pi / 2)
    delta = torch.zeros_like(root[:, 3:7])
    delta[:, 0], delta[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    root[:, 3:7] = quat_mul(delta, root[:, 3:7])
    root[:, :3] = env.scene.env_origins[ids]
    up = torch.zeros_like(root[:, :3])
    up[:, 2] = 1
    local_up = quat_apply_inverse(root[:, 3:7], up)
    for mask, pose in ((~snake, tuple(DEFAULT_JOINT_POS)), (snake, (0.,) * 6)):
        radial, low, high = collision_radial_envelope(pose)
        zmin = (local_up[mask, :2] @ root.new_tensor(radial).T).amin(-1)
        zmin += torch.minimum(local_up[mask, 2] * low, local_up[mask, 2] * high)
        root[mask, 2] += .003 - zmin
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
    robot.write_root_state_to_sim(root, env_ids=ids)


class UniversalOmniEvaluator:
    """Twenty independent scenarios with four initial headings and real keyboard changes."""

    def __init__(self, device='cpu', num_envs=80, seconds=10., env_cfg=None, randomized=True, **_legacy_settings):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError('num_envs must be a positive integer')
        if not math.isfinite(seconds) or seconds < 2:
            raise ValueError('evaluation requires at least two seconds')
        self.device = torch.device(device)
        self.num_envs = max(80, math.ceil(num_envs / 80) * 80)
        self.seconds = seconds
        self.env_cfg = deepcopy(env_cfg)
        self.randomized = randomized
        self.rl_iteration = 0
        self.foundation_samples = None
        self.env = None
        self.wrapper = None

    def _build(self):
        cfg = evaluation_env_cfg(self.env_cfg, num_envs=self.num_envs, randomized=self.randomized)
        cfg.scene.num_envs = self.num_envs
        cfg.events['reset_transition'] = EventTermCfg(func=reset_omni_evaluation_pose, mode='reset')
        self.env = ManagerBasedRlEnv(cfg, device=str(self.device))
        self.wrapper = RslRlVecEnvWrapper(self.env)

    def _commands(self, command, *, initial):
        width = self.num_envs // len(SCENARIOS)
        for index, (name, mode, velocity) in enumerate(SCENARIOS):
            ids = torch.arange(index * width, (index + 1) * width, device=self.device)
            target = 'stand' if mode == 'biped' else 'snake'
            if initial and name in ('ascent', 'descent'):
                target = 'snake' if name == 'ascent' else 'stand'
            command.set_target(target, ids)
            if initial and getattr(command.cfg, 'initialize_matched_motion', False):
                # These trials explicitly reset to the requested canonical pose.
                # Start motion directly, just as the teacher's matched training reset.
                command.initialize_task_state(ids, torch.full((len(ids),), target == 'stand', device=self.device, dtype=torch.bool))
            # Stop trials first move forward, then receive a keyboard stop.
            request = ((.08, 0., 0.) if name.endswith('_stop') else (0., 0., 0.)) if initial else velocity
            command.set_velocity(*request, ids)

    def __call__(self, actor):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        training, python_rng, numpy_rng = actor.training, random.getstate(), np.random.get_state()
        self.foundation_samples = None
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
                physical = cast(Any, env)
                physical._student_rl_iteration = self.rl_iteration
                physical._student_pretraining = False
                wrapper.reset()
                command = mdp.command_term(env)
                self._commands(command, initial=True)
                env.observation_manager.reset()
                env.observation_manager.compute(update_history=True, env_ids=torch.arange(self.num_envs, device=self.device))
                obs = wrapper.get_observations()
                velocities, ready = [], []
                finite = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
                no_fall = torch.ones_like(finite)
                group = torch.arange(self.num_envs, device=self.device) // (self.num_envs // len(SCENARIOS))
                biped_motion = group < len(MOTIONS)
                switch = math.ceil(1. / env.step_dt)
                steps = switch + math.ceil(self.seconds / env.step_dt)
                measurement = switch + (steps - switch) // 2
                for step in range(steps):
                    if step == switch:
                        self._commands(command, initial=False)
                    actions = actor(obs, stochastic_output=False)
                    obs, rewards, _, _ = wrapper.step(actions)
                    finite &= torch.isfinite(obs['actor']).all(-1) & torch.isfinite(rewards) & ~mdp.unsafe(env)
                    no_fall &= ~biped_motion | (mdp.base_height(env) >= .10)
                    if step >= measurement:
                        velocities.append(mdp.tracking_velocity(env).clone())
                        ready.append(mdp.configuration_ready(env).clone())
                return evaluate_omni_capabilities(torch.stack(velocities), torch.stack(ready), finite, no_fall=no_fall)
        finally:
            actor.train(training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = self.wrapper = None
        self.foundation_samples = None
