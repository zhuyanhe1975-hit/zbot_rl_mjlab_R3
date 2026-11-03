"""Synchronous PBT: independent PPO learners share one batched simulation.

No demonstrations, reset curriculum, or physical parameter mutation. Each learner
owns a disjoint world slice and its own models, optimizer and rollout storage.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from rsl_rl.algorithms import PPO

from .tasks.reconfiguration import mdp


INITIAL_PROFILES = (
    (.003, .0003, .995, .3), (.01, .001, .995, .6),
    (.06, .0003, .995, 1.2), (.01, .0003, .999, .6),
    (.03, .001, .99, .9), (.03, .0001, .995, .9),
    (.01, .001, .995, .6), (.08, .0005, .999, 1.4),
)


@dataclass
class PbtCfg:
    enabled: bool = True
    population_size: int = 8
    # Each row: entropy coefficient, learning rate, gamma, initial action std.
    initial_profiles: tuple[tuple[float, float, float, float], ...] = INITIAL_PROFILES
    focused_population_size: int = 4
    focus_after_iterations: int = 100
    warmup_iterations: int = 100
    interval_iterations: int = 100
    evaluation_envs: int = 16
    # None uses the complete task episode; short values are only for smoke tests.
    evaluation_episode_s: float | None = None
    mutation_probability: float = .5
    mutation_factor: float = 1.5
    minimum_score_gap: float = .02
    protect_explorer: bool = True
    entropy_bounds: tuple[float, float] = (.001, .1)
    learning_rate_bounds: tuple[float, float] = (1e-5, .003)
    gamma_bounds: tuple[float, float] = (.97, .9995)
    std_bounds: tuple[float, float] = (.15, 1.5)


@dataclass
class ReconfigurationRunnerCfg(RslRlOnPolicyRunnerCfg):
    pbt: PbtCfg = field(default_factory=PbtCfg)


def validate(cfg, num_envs):
    for name in ('population_size', 'warmup_iterations', 'interval_iterations', 'evaluation_envs'):
        if type(cfg[name]) is not int or cfg[name] < (2 if name == 'population_size' else 1):
            raise ValueError(f'pbt.{name}: invalid positive integer')
    if (type(cfg['focused_population_size']) is not int or not 2 <= cfg['focused_population_size'] <= cfg['population_size']
            or num_envs % cfg['focused_population_size']):
        raise ValueError('focused_population_size must divide total worlds and fit initial population')
    if type(cfg['focus_after_iterations']) is not int or cfg['focus_after_iterations'] < 1:
        raise ValueError('focus_after_iterations must be positive')
    if num_envs % cfg['population_size'] or num_envs < cfg['population_size']:
        raise ValueError('total num_envs must be divisible by pbt.population_size')
    if not 0 <= cfg['mutation_probability'] <= 1 or not math.isfinite(cfg['mutation_probability']):
        raise ValueError('invalid mutation_probability')
    if not math.isfinite(cfg['mutation_factor']) or cfg['mutation_factor'] <= 1:
        raise ValueError('mutation_factor must exceed 1')
    if not math.isfinite(cfg['minimum_score_gap']) or cfg['minimum_score_gap'] < 0:
        raise ValueError('minimum_score_gap must be nonnegative')
    for name in ('entropy_bounds', 'learning_rate_bounds', 'gamma_bounds', 'std_bounds'):
        lo, hi = cfg[name]
        if not 0 < lo <= hi or not all(map(math.isfinite, (lo, hi))):
            raise ValueError(f'invalid {name}')
    if cfg['gamma_bounds'][1] >= 1:
        raise ValueError('gamma must be below one')
    profiles = cfg['initial_profiles']
    if len(profiles) < 2:
        raise ValueError('initial_profiles requires at least two profiles')
    for row in profiles:
        if len(row) != 4:
            raise ValueError('profile order: entropy, learning_rate, gamma, std')
        for value, key in zip(row, ('entropy_bounds', 'learning_rate_bounds', 'gamma_bounds', 'std_bounds'), strict=True):
            if not math.isfinite(value) or not cfg[key][0] <= value <= cfg[key][1]:
                raise ValueError(f'initial profile outside {key}')
    duration = cfg['evaluation_episode_s']
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError('evaluation_episode_s must be positive')


def objective(metrics):
    """Success dominates all bounded auxiliary terms, never use shaped return."""
    return (100. * metrics['success_rate'] + .5 * metrics['pose_quality']
            + .1 * metrics['coverage'] - .1 * metrics['failure_rate'])


def choose_replacement(scores, cfg):
    """Replace at most one weak member; keep an independent explorer lineage."""
    best = max(range(len(scores)), key=scores.__getitem__)
    candidates = [i for i in range(len(scores)) if i != best
                  and not (cfg['protect_explorer'] and i == len(scores) - 1)]
    if not candidates:
        return None
    worst = min(candidates, key=scores.__getitem__)
    if scores[best] - scores[worst] <= cfg['minimum_score_gap']:
        return None
    return worst, best


def mutate(parameters, cfg, rng):
    result = dict(parameters)
    for key, bounds_key in (('entropy', 'entropy_bounds'), ('learning_rate', 'learning_rate_bounds'),
                            ('gamma', 'gamma_bounds'), ('std', 'std_bounds')):
        if rng.random() >= cfg['mutation_probability']:
            continue
        factor = cfg['mutation_factor'] ** rng.choice((-1, 1))
        value = 1 - (1 - result[key]) * factor if key == 'gamma' else result[key] * factor
        lo, hi = cfg[bounds_key]
        result[key] = min(hi, max(lo, value))
    return result


def parameters(alg):
    distribution = alg.get_policy().distribution
    if hasattr(distribution, 'std_param'):
        std = distribution.std_param.detach().clamp_min(1e-6).mean().item()
    else:
        std = distribution.log_std_param.detach().exp().mean().item()
    return dict(entropy=alg.entropy_coef, learning_rate=alg.learning_rate, gamma=alg.gamma, std=std)


def apply_parameters(alg, hp, reset_std_momentum=True, std_bounds=(.15, 1.5)):
    alg.entropy_coef, alg.learning_rate, alg.gamma = hp['entropy'], hp['learning_rate'], hp['gamma']
    for group in alg.optimizer.param_groups:
        group['lr'] = hp['learning_rate']
    if not reset_std_momentum:
        return  # weights already contain the exact saved std; preserve optimizer moments
    distribution = alg.get_policy().distribution
    with torch.no_grad():
        if hasattr(distribution, 'std_param'):
            value = distribution.std_param
            value.mul_(hp['std'] / value.clamp_min(1e-6).mean()).clamp_(*std_bounds)
        else:
            value = distribution.log_std_param
            value.add_(math.log(hp['std'] / value.exp().mean().item())).clamp_(math.log(std_bounds[0]), math.log(std_bounds[1]))
    # Old optimizer momentum must not undo the explicit exploration mutation.
    param = getattr(distribution, 'std_param', getattr(distribution, 'log_std_param', None))
    if reset_std_momentum:
        alg.optimizer.state.pop(param, None)


class _PopulationSlice:
    """Construction adapter only; the coordinator steps the entire batch once."""
    def __init__(self, env, selection):
        self.env, self.selection = env, selection
        self.num_envs = selection.stop - selection.start

    def get_observations(self):
        return self.env.get_observations()[self.selection]

    def __getattr__(self, name):
        return getattr(self.env, name)


class ReconfigurationPbtRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.pbt = copy.deepcopy(train_cfg.get('pbt', {}))
        self.population_mode = bool(self.pbt.get('enabled', False) and 'success' in env.cfg.terminations)
        if not self.population_mode:
            super().__init__(env, train_cfg, log_dir, device)
            return
        if int(os.getenv('WORLD_SIZE', '1')) != 1:
            raise ValueError('PBT uses one simulation batch; distributed PPO is not supported')
        validate(self.pbt, env.num_envs)
        count = self.pbt['population_size']
        width = env.num_envs // count
        self.slices = [slice(i * width, (i + 1) * width) for i in range(count)]
        # Use the normal MJLab construction path for configuration migration.
        super().__init__(_PopulationSlice(env, self.slices[0]), copy.deepcopy(train_cfg), log_dir, device)
        self.members = [self.alg]
        for i in range(1, count):
            torch.manual_seed(train_cfg['seed'] + 1009 * i)
            member_cfg = copy.deepcopy(self.cfg)
            self.members.append(PPO.construct_algorithm(env.get_observations()[self.slices[i]],
                                _PopulationSlice(env, self.slices[i]), member_cfg, device))
        self.env = env
        self.rng = random.Random(train_cfg['seed'])
        self.best_member = 0
        self.last_evaluation = None
        self._last_eval_iteration = -1
        self.evaluation_env = None
        self.log_path = Path(log_dir) if log_dir else None
        self.writer = None
        self.total_env_steps = 0
        self.next_iteration = 0
        profiles = self.pbt['initial_profiles']
        for i, alg in enumerate(self.members):
            values = profiles[round(i * (len(profiles) - 1) / (count - 1))]
            apply_parameters(alg, dict(zip(('entropy', 'learning_rate', 'gamma', 'std'), values, strict=True)),
                             std_bounds=self.pbt['std_bounds'])
        self.focused = False
        self.best_score = -math.inf
        print(f'PBT: {count} independent policies x {width} worlds = {env.num_envs} total worlds')

    def _event(self, event):
        event = {'iteration': self.next_iteration, 'total_env_steps': self.total_env_steps, **event}
        if self.log_path:
            with (self.log_path / 'pbt_events.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
        print('PBT', json.dumps(event, allow_nan=False))

    def _evaluate(self):
        width = self.pbt['evaluation_envs']
        total = width * len(self.members)
        if self.evaluation_env is not None and self.evaluation_env.num_envs != total:
            self.evaluation_env.close()
            self.evaluation_env = None
        if self.evaluation_env is None:
            cfg = copy.deepcopy(self.env.cfg)
            cfg.scene.num_envs = total
            cfg.seed = self.cfg['seed'] + 500_000
            if self.pbt['evaluation_episode_s'] is not None:
                cfg.episode_length_s = self.pbt['evaluation_episode_s']
            self.evaluation_env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=str(self.env.device)),
                                                    clip_actions=self.cfg.get('clip_actions'))
        wrapped = self.evaluation_env
        core = wrapped.unwrapped
        selections = [slice(i * width, (i + 1) * width) for i in range(len(self.members))]
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            # Independent evaluation batch; neither training worlds nor RNG streams change.
            with torch.random.fork_rng(devices=[self.env.device] if self.env.device.type == 'cuda' else []):
                wrapped.seed(self.cfg['seed'] + 500_000)
                obs, _ = wrapped.reset()
                # Every population member gets exactly the same goal distribution.
                goal = core.command_manager.get_term('dock')
                balanced = goal.choices[torch.arange(width, device=wrapped.device) % len(goal.choices)]
                goal.quarter.copy_(balanced.repeat(len(self.members)))
                obs = wrapped.get_observations()
                for alg in self.members:
                    alg.eval_mode()
                active = torch.ones(total, device=wrapped.device, dtype=torch.bool)
                success = torch.zeros_like(active)
                failed = torch.zeros_like(active)
                quality = torch.zeros(total, device=wrapped.device)
                descriptor = torch.zeros(total, 6, device=wrapped.device)
                visited = []
                for step in range(core.max_episode_length + 1):
                    dock = mdp.state(core)
                    g = dock.geometry()
                    # Position and orientation must improve at the same instant.
                    pose = torch.exp(-g['distance'] / .2 - g['angle'] / .5)
                    quality = torch.maximum(quality, torch.where(active, pose, 0.))
                    descriptor[active] = dock.robot.data.joint_pos[active]
                    if step % 10 == 0:
                        # Center a bin on zero so tiny initial jitter is not novelty.
                        bins = (dock.robot.data.joint_pos / (math.pi / 4)).round().long().clamp(-4, 4) + 4
                        code = (bins * bins.new_tensor([9**i for i in range(6)])).sum(-1)
                        visited.append(torch.where(active, code, -1))
                    actions = torch.cat([alg.get_policy()(obs[selection].to(self.device))
                        for alg, selection in zip(self.members, selections, strict=True)])
                    obs, reward, done, _ = wrapped.step(actions.to(wrapped.device))
                    if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                        raise FloatingPointError('nonfinite PBT evaluation')
                    success |= active & core.termination_manager.get_term('success')
                    failed |= active & core.termination_manager.get_term('unsafe')
                    active &= ~done.bool()
                    if not active.any():
                        break
                codes = torch.stack(visited)
                results = []
                for selection in selections:
                    group = codes[:, selection]
                    valid = group[group >= 0]
                    coverage = min(1., valid.unique().numel() / max(1, valid.numel()))
                    metrics = dict(success_rate=success[selection].float().mean().item(),
                        failure_rate=failed[selection].float().mean().item(),
                        pose_quality=quality[selection].mean().item(), coverage=coverage,
                        descriptor=(descriptor[selection] / math.pi).mean(0).tolist())
                    metrics['score'] = objective(metrics)
                    results.append(metrics)
                return results
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            for alg in self.members:
                alg.train_mode()

    def _select(self, results):
        self._last_eval_iteration = self.next_iteration
        scores = [row['score'] for row in results]
        self.best_member = max(range(len(scores)), key=scores.__getitem__)
        self.alg = self.members[self.best_member]
        self.last_evaluation = results
        if self.writer:
            for i, row in enumerate(results):
                for name in ('score', 'success_rate', 'failure_rate', 'pose_quality', 'coverage'):
                    self.writer.add_scalar(f'member_{i}/evaluation/{name}', row[name], self.next_iteration)
        self._event({'event': 'evaluation', 'members': results, 'best_member': self.best_member})
        if scores[self.best_member] > self.best_score:
            self.best_score = scores[self.best_member]
            if self.log_path:
                # Preserve exactly the evaluated policy before any mutations.
                self.save(str(self.log_path / 'best.pt'))
        if self.next_iteration < self.pbt['warmup_iterations']:
            return  # A final evaluation of a short run must not bypass exploration warmup.
        if (not self.focused and self.next_iteration >= self.pbt['focus_after_iterations']
                and len(self.members) > self.pbt['focused_population_size']
                and max(scores) - min(scores) > self.pbt['minimum_score_gap']):
            self._focus(results)
            return
        choice = choose_replacement(scores, self.pbt)
        if choice is None:
            return
        target, source = choice
        donor = copy.deepcopy(self.members[source].save())
        receiver = self.members[target]
        receiver.load(donor, None, True)  # actor, critic, normalizers, optimizer
        hp = mutate(parameters(self.members[source]), self.pbt, self.rng)
        apply_parameters(receiver, hp, std_bounds=self.pbt['std_bounds'])
        receiver.storage.clear()
        receiver.transition.clear()
        # Recipient physics and integrated targets must restart together.
        selection = self.slices[target]
        ids = torch.arange(selection.start, selection.stop, device=self.env.device)
        with torch.inference_mode():
            self.env.unwrapped._reset_idx(ids)
            self.env.unwrapped.sim.forward()
            self.env.unwrapped.scene.update(self.env.unwrapped.physics_dt)
        self._event({'event': 'replacement', 'source': source, 'target': target, 'parameters': hp})

    def _focus(self, results):
        count = self.pbt['focused_population_size']
        scores = [row['score'] for row in results]
        selected = [self.best_member]
        explorer = len(self.members) - 1
        slots = count - int(self.pbt['protect_explorer'] and explorer != self.best_member)
        while len(selected) < slots:
            candidates = [i for i in range(len(self.members)) if i not in selected
                          and not (self.pbt['protect_explorer'] and i == explorer)]
            def priority(i):
                distance = min(math.dist(results[i]['descriptor'], results[j]['descriptor']) for j in selected)
                return scores[i] + .1 * distance / math.sqrt(6)
            selected.append(max(candidates, key=priority))
        if self.pbt['protect_explorer'] and explorer not in selected:
            selected.append(explorer)
        # Keep the protected independent lineage last after the population shrinks.
        if self.pbt['protect_explorer']:
            selected = [i for i in selected if i != explorer] + [explorer]
        old_members = self.members
        width = self.env.num_envs // count
        self.slices = [slice(i * width, (i + 1) * width) for i in range(count)]
        self.members = []
        for source, selection in zip(selected, self.slices, strict=True):
            saved = copy.deepcopy(old_members[source].save())
            hp = parameters(old_members[source])
            alg = PPO.construct_algorithm(self.env.get_observations()[selection],
                    _PopulationSlice(self.env, selection), copy.deepcopy(self.cfg), self.device)
            alg.load(saved, None, True)
            apply_parameters(alg, hp, reset_std_momentum=False)
            alg.train_mode()
            self.members.append(alg)
        self.best_member = selected.index(self.best_member)
        self.alg = self.members[self.best_member]
        self.last_evaluation = [results[i] for i in selected]
        self.focused = True
        with torch.inference_mode():
            self.env.reset()
        self._event({'event': 'focus', 'selected_original_members': selected,
                     'population_size': count, 'envs_per_member': width})

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if not self.population_mode:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        from torch.utils.tensorboard import SummaryWriter
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_path))
            self.logger._store_code_state()
        with torch.inference_mode():
            obs, _ = self.env.reset()
        for alg in self.members:
            alg.train_mode()
        stop = self.next_iteration + num_learning_iterations
        try:
            for iteration in range(self.next_iteration, stop):
                started = time.perf_counter()
                rollout_rewards = torch.zeros(len(self.members), device=self.env.device)
                rollout_successes = torch.zeros_like(rollout_rewards)
                rollout_failures = torch.zeros_like(rollout_rewards)
                term_rewards = torch.zeros(len(self.members), len(self.env.unwrapped.reward_manager.active_terms), device=self.env.device)
                with torch.inference_mode():
                    for _ in range(self.cfg['num_steps_per_env']):
                        actions = torch.cat([alg.act(obs[selection].to(self.device))
                                             for alg, selection in zip(self.members, self.slices, strict=True)])
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        if not torch.isfinite(rewards).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                            raise FloatingPointError('nonfinite PBT rollout')
                        for member, (alg, selection) in enumerate(zip(self.members, self.slices, strict=True)):
                            rollout_rewards[member] += rewards[selection].mean()
                            rollout_successes[member] += self.env.unwrapped.termination_manager.get_term('success')[selection].sum()
                            rollout_failures[member] += self.env.unwrapped.termination_manager.get_term('unsafe')[selection].sum()
                            term_rewards[member] += self.env.unwrapped.reward_manager._step_reward[selection].mean(0)
                            info = {'time_outs': extras['time_outs'][selection].to(self.device)} if 'time_outs' in extras else {}
                            alg.process_env_step(obs[selection].to(self.device), rewards[selection].to(self.device),
                                                 dones[selection].to(self.device), info)
                    for alg, selection in zip(self.members, self.slices, strict=True):
                        alg.compute_returns(obs[selection].to(self.device))
                losses = [alg.update() for alg in self.members]
                self.total_env_steps += self.env.num_envs * self.cfg['num_steps_per_env']
                self.next_iteration = iteration + 1
                self.current_learning_iteration = iteration
                for i, (alg, loss) in enumerate(zip(self.members, losses, strict=True)):
                    if self.writer:
                        stats = dict(mean_step_reward=rollout_rewards[i].item() / self.cfg['num_steps_per_env'],
                                     success_episodes=rollout_successes[i].item(), unsafe_episodes=rollout_failures[i].item())
                        for key, value in {**loss, **parameters(alg), **stats}.items():
                            self.writer.add_scalar(f'member_{i}/{key}', value, self.next_iteration)
                        for j, name in enumerate(self.env.unwrapped.reward_manager.active_terms):
                            self.writer.add_scalar(f'member_{i}/reward/{name}', term_rewards[i, j].item() / self.cfg['num_steps_per_env'], self.next_iteration)
                if self.next_iteration % 10 == 0 or self.next_iteration == stop:
                    self._event({'event': 'progress', 'iteration_seconds': time.perf_counter() - started})
                if self.next_iteration >= self.pbt['warmup_iterations'] and (
                        (self.next_iteration - self.pbt['warmup_iterations']) % self.pbt['interval_iterations'] == 0):
                    with torch.no_grad():
                        self._select(self._evaluate())
                        obs = self.env.get_observations()
                if self.log_path and self.next_iteration % self.cfg['save_interval'] == 0:
                    self.save(str(self.log_path / f'model_{iteration}.pt'))
                    self.save_population(self.log_path / 'population.pt')
            if self.last_evaluation is None or self.next_iteration != self._last_eval_iteration:
                with torch.no_grad():
                    self._select(self._evaluate())
            if self.log_path:
                self.save(str(self.log_path / f'model_{self.current_learning_iteration}.pt'))
                self.save_population(self.log_path / 'population.pt')
        finally:
            if self.writer:
                self.writer.close()
            if self.evaluation_env is not None:
                self.evaluation_env.close()
                self.evaluation_env = None

    def save(self, path, infos=None):
        if not self.population_mode:
            return super().save(path, infos)
        payload = self.members[self.best_member].save()
        payload.update(iter=self.current_learning_iteration, infos=infos,
                       pbt={'member': self.best_member, 'next_iteration': self.next_iteration,
                            'evaluation': self.last_evaluation, 'parameters': parameters(self.members[self.best_member])})
        torch.save(payload, path)

    def save_population(self, path):
        payload = dict(pbt_population_version=1, members=[alg.save() for alg in self.members],
                       parameters=[parameters(alg) for alg in self.members], next_iteration=self.next_iteration,
                       total_env_steps=self.total_env_steps, best_member=self.best_member,
                       evaluation=self.last_evaluation, random_state=self.rng.getstate(),
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                       pbt=self.pbt, focused=self.focused, best_score=self.best_score,
                       last_eval_iteration=self._last_eval_iteration)
        torch.save(payload, path)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if not self.population_mode:
            return super().load(path, load_cfg, strict, map_location)
        payload = torch.load(path, map_location=map_location or self.device, weights_only=False)
        if payload.get('pbt_population_version') != 1:
            raise ValueError('Resume PBT with population.pt; use --agent.pbt.enabled False for a single-policy checkpoint')
        if payload['pbt'] != self.pbt:
            raise ValueError('PBT resume requires the same population and PBT configuration')
        count = len(payload['members'])
        expected = self.pbt['focused_population_size'] if payload['focused'] else self.pbt['population_size']
        if count != expected:
            raise ValueError('invalid saved population size')
        if count != len(self.members):
            width = self.env.num_envs // count
            self.slices = [slice(i * width, (i + 1) * width) for i in range(count)]
            self.members = [PPO.construct_algorithm(self.env.get_observations()[selection],
                _PopulationSlice(self.env, selection), copy.deepcopy(self.cfg), self.device) for selection in self.slices]
            self.alg = self.members[0]
        for alg, saved, hp in zip(self.members, payload['members'], payload['parameters'], strict=True):
            alg.load(saved, load_cfg, strict)
            apply_parameters(alg, hp, reset_std_momentum=False)
            alg.storage.clear()
            alg.transition.clear()
        self.focused = payload['focused']
        self.best_score = payload['best_score']
        self._last_eval_iteration = payload['last_eval_iteration']
        self.next_iteration = payload['next_iteration']
        self.current_learning_iteration = self.next_iteration - 1
        self.total_env_steps = payload['total_env_steps']
        self.best_member = payload['best_member']
        self.alg = self.members[self.best_member]
        self.last_evaluation = payload['evaluation']
        self.rng.setstate(payload['random_state'])
        torch.set_rng_state(payload['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_rng']])
        return {}
