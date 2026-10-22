"""Independent fixed-command acceptance for student PPO checkpoints."""

import math
import random
from typing import Any, cast

import numpy as np
import torch

from .tasks.universal import mdp
from .tasks.universal.guided_transitions import GuidedOmniCommand
from .tasks.walking import mdp as walking
from .universal_omni_evaluation import (
    MOTIONS,
    SCENARIOS,
    UniversalOmniEvaluator,
    evaluate_omni_capabilities,
)


def summarize_student_traces(velocity, ready, finite, *, no_fall=None):
    """Report signed cases separately so opposite directions cannot cancel."""
    metrics = evaluate_omni_capabilities(velocity, ready, finite, no_fall=no_fall)
    width = velocity.shape[1] // len(SCENARIOS)
    for index, (name, _, target) in enumerate(SCENARIOS):
        actual = velocity[:, index * width:(index + 1) * width]
        error = actual - actual.new_tensor(target)
        for axis, label in enumerate(('vx', 'vy', 'wz')):
            metrics[f'{name}_rmse_{label}'] = float(error[..., axis].square().mean().sqrt())
            metrics[f'{name}_mean_{label}'] = float(actual[..., axis].mean())
        moving = actual.norm(dim=-1) > .01
        metrics[f'{name}_moving_fraction'] = float(moving.float().mean())
    for mode in ('biped', 'snake'):
        names = [f'{mode}_{name}' for name, _ in MOTIONS if name != 'stop']
        metrics[f'{mode}_moving_success'] = sum(metrics[f'{name}_success'] for name in names) / len(names)
        mixed = [f'{mode}_mixed_positive', f'{mode}_mixed_negative']
        metrics[f'{mode}_worst_mixed_success'] = min(metrics[f'{name}_success'] for name in mixed)
    return metrics


class StudentPPOEvaluator(UniversalOmniEvaluator):
    """Reuse canonical scenarios, adding errors, handoff times and foot quality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.env_cfg is not None:
            self.env_cfg.student_observation_noise = False

    def __call__(self, actor):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        training, python_rng, numpy_rng = actor.training, random.getstate(), np.random.get_state()
        try:
            actor.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                if self.device.type == 'cuda':
                    torch.cuda.manual_seed(42)
                if self.env is None:
                    self._build()
                env, wrapper = self.env, self.wrapper
                assert env is not None and wrapper is not None
                physical = cast(Any, env)
                physical._student_rl_iteration = self.rl_iteration
                physical._student_pretraining = False
                wrapper.reset()
                command = cast(GuidedOmniCommand, mdp.command_term(env))
                self._commands(command, initial=True)
                env.observation_manager.reset()
                env.observation_manager.compute(update_history=True, env_ids=torch.arange(self.num_envs, device=self.device))
                obs = wrapper.get_observations()
                width = self.num_envs // len(SCENARIOS)
                group = torch.arange(self.num_envs, device=self.device) // width
                biped = group < len(MOTIONS)
                finite = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
                no_fall = torch.ones_like(finite)
                handoff = torch.full((self.num_envs,), float('nan'), device=self.device)
                peak_speed = torch.zeros(self.num_envs, device=self.device)
                velocities, readiness, landing_angles = [], [], []
                switch = math.ceil(1. / env.step_dt)
                steps = switch + math.ceil(self.seconds / env.step_dt)
                measurement = switch + (steps - switch) // 2
                for step in range(steps):
                    if step == switch:
                        self._commands(command, initial=False)
                    actions = actor(obs, stochastic_output=False)
                    obs, rewards, _, _ = wrapper.step(actions)
                    finite &= torch.isfinite(actions).all(-1) & torch.isfinite(obs['actor']).all(-1) & torch.isfinite(rewards) & ~mdp.unsafe(env)
                    no_fall &= ~biped | (mdp.base_height(env) >= .10)
                    if step >= switch:
                        elapsed = (step - switch + 1) * env.step_dt
                        for name, task in (('descent', 2), ('ascent', 0)):
                            index = next(i for i, item in enumerate(SCENARIOS) if item[0] == name)
                            selected = (group == index) & (command.current_task == task) & torch.isnan(handoff)
                            handoff[selected] = elapsed
                        peak_speed = torch.maximum(peak_speed, env.scene['robot'].data.joint_vel.abs().amax(-1))
                        first = env.scene['feet_ground_contact'].compute_first_contact(dt=env.step_dt)
                        angle = torch.rad2deg(torch.asin(walking.sole_tilt_sq(env).sqrt().clamp(0, 1)))
                        mask = biped[:, None] & (command.current_task == 0)[:, None] & first
                        landing_angles.append(angle[mask].clone())
                    if step >= measurement:
                        velocities.append(mdp.tracking_velocity(env).clone())
                        readiness.append(mdp.configuration_ready(env).clone())
                metrics = summarize_student_traces(torch.stack(velocities), torch.stack(readiness), finite, no_fall=no_fall)
                for index, (name, _, _) in enumerate(SCENARIOS):
                    selected = group == index
                    metrics[f'{name}_peak_joint_speed'] = float(peak_speed[selected].max())
                    if name in ('ascent', 'descent'):
                        reached = handoff[selected]
                        valid = torch.isfinite(reached)
                        metrics[f'{name}_handoff_fraction'] = float(valid.float().mean())
                        metrics[f'{name}_handoff_mean_s'] = float(reached[valid].mean()) if valid.any() else 0.
                angles = torch.cat(landing_angles) if landing_angles else peak_speed[:0]
                metrics['landing_events'] = float(angles.numel())
                metrics['landing_angle_mean_deg'] = float(angles.mean()) if angles.numel() else 0.
                metrics['landing_angle_p95_deg'] = float(angles.quantile(.95)) if angles.numel() else 0.
                return metrics
        finally:
            actor.train(training)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
