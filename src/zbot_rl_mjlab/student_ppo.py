"""PPO fine tuning with a frozen supervised student motion reference.

The archived teacher is neither constructed nor queried. Cold transfer imports
only the DAgger actor; PPO checkpoints have their own explicit resume contract.
"""

import copy
import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlPpoAlgorithmCfg
from tensordict import TensorDict

from .gaussian_action_models import GaussianTaskDistribution
from .grouped_ppo import GroupedRolloutStorage
from .student import TEACHER_SHA256, StudentRunnerCfg
from .student_metrics import StudentTaskMetrics
from .student_models import HistoryGaussianStudentMLP, HistoryTaskRoutedNetwork
from .task_ppo import TaskPPO
from .tasks.universal.student_observations import GuidedStudentEnvCfg


@dataclass
class StudentPPOCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'zbot_rl_mjlab.student_ppo:StudentPPO'
    critic_warmup_iterations: int = 10
    reference_tasks: tuple[int, ...] = (0, 2)
    reference_start_coef: float = .05
    reference_floor_coef: float = .005
    reference_decay_iterations: int = 200
    reference_batch_size: int = 1024
    reference_learning_rate: float = 1e-3
    task_balanced_advantages: bool = False


@dataclass
class StudentPPORunnerCfg(StudentRunnerCfg):
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=StudentPPOCfg)
    initial_student_checkpoint: str | None = None
    initial_student_sha256: str | None = None
    console_mode: str = 'compact'
    validation_num_envs: int = 80
    validation_seconds: float = 8.
    validation_selection_enabled: bool = False
    validation_regression_tolerance: float = .25
    validation_regression_patience: int = 3


class PPOHistoryGaussianStudentMLP(HistoryGaussianStudentMLP):
    """Preserve the supervised input coordinate system during fine tuning."""

    def update_normalization(self, obs):
        pass


def _actor_only(load_cfg):
    return load_cfg is not None and load_cfg.get('actor', False) and not any(
        load_cfg.get(key, False) for key in ('critic', 'optimizer', 'iteration', 'rnd')
    )


class StudentPPO(TaskPPO):
    """Task-routed PPO with critic warmup and bounded motion retention."""

    def __init__(self, *args, critic_warmup_iterations=10, reference_tasks=(0, 2),
                 reference_start_coef=.05, reference_floor_coef=.005,
                 reference_decay_iterations=200, reference_batch_size=1024,
                 reference_learning_rate=1e-3, task_balanced_advantages=False, **kwargs):
        if type(critic_warmup_iterations) is not int or critic_warmup_iterations < 0:
            raise ValueError('critic_warmup_iterations must be a nonnegative integer')
        if (not isinstance(reference_tasks, (tuple, list)) or not reference_tasks
                or len(set(reference_tasks)) != len(reference_tasks)
                or any(type(task) is not int or task not in range(4) for task in reference_tasks)):
            raise ValueError('reference_tasks must contain unique task IDs from 0 through 3')
        if type(task_balanced_advantages) is not bool:
            raise ValueError('task_balanced_advantages must be a boolean')
        if (not all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (reference_start_coef, reference_floor_coef))
                or not 0 <= reference_floor_coef <= reference_start_coef):
            raise ValueError('reference coefficients require 0 <= floor <= start and finite values')
        if (type(reference_learning_rate) not in (int, float)
                or not math.isfinite(reference_learning_rate) or reference_learning_rate <= 0):
            raise ValueError('reference_learning_rate must be finite and positive')
        for name, value in (('reference_decay_iterations', reference_decay_iterations),
                            ('reference_batch_size', reference_batch_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        super().__init__(*args, **kwargs)
        self.reference_tasks = tuple(reference_tasks)
        self.reference_start_coef = reference_start_coef
        self.reference_floor_coef = reference_floor_coef
        self.reference_decay_iterations = reference_decay_iterations
        self.reference_batch_size = reference_batch_size
        self.reference_learning_rate = reference_learning_rate
        self.reference_updates = 0
        self.task_balanced_advantages = task_balanced_advantages
        self._rollout_metrics = {}
        self.initialize_reference()
        self.critic_warmup_iterations = critic_warmup_iterations
        self.completed_updates = 0

    def initialize_reference(self):
        """Snapshot the transferred student, including its frozen input moments."""
        self.reference_actor = copy.deepcopy(self._raw_actor).eval().requires_grad_(False)

    def compute_returns(self, obs):
        # TaskPPO already computes task-local GAE and normalizes each task.
        # Balance only the policy objective; preserve critic targets and GAE.
        super().compute_returns(obs)
        storage = cast(GroupedRolloutStorage, self.storage)
        counts = storage.group_active_masks.sum(dim=(0, 1))
        total = counts.sum()
        active_count = (counts > 0).sum().clamp_min(1)
        weights = total / (active_count * counts.clamp_min(1))
        if self.task_balanced_advantages:
            storage.advantages.mul_((storage.group_active_masks * weights).sum(-1, keepdim=True))
        self._rollout_metrics = {}
        for task, count in enumerate(counts):
            prefix = f'student_ppo_task{task}'
            self._rollout_metrics[f'{prefix}_sample_fraction'] = float(count / total.clamp_min(1))
            if count == 0:
                continue
            mask = storage.group_active_masks[..., task].bool()
            returns = storage.returns[..., task][mask]
            values = storage.values[..., task][mask]
            advantage = returns - values
            self._rollout_metrics.update({
                f'{prefix}_return_mean': float(returns.mean()),
                f'{prefix}_value_mean': float(values.mean()),
                f'{prefix}_advantage_mean': float(advantage.mean()),
                f'{prefix}_advantage_std': float(advantage.std(unbiased=False)),
                f'{prefix}_policy_advantage_std': float(storage.advantages[..., 0][mask].std(unbiased=False)),
                f'{prefix}_policy_weight': float(weights[task]) if self.task_balanced_advantages else 1.,
            })

    @torch.no_grad()
    def _ppo_kl_metrics(self):
        """Old rollout Gaussian -> post-PPO Gaussian, before reference correction.

        Use at most 1024 evenly spaced frames per task, without drawing random
        actions or changing the RNG stream (including for legacy v2 runs).
        """
        storage = cast(GroupedRolloutStorage, self.storage)
        if storage.distribution_params is None:
            return {}
        observations = cast(TensorDict, storage.observations.flatten(0, 1))
        masks = storage.group_active_masks.flatten(0, 1)
        old = tuple(value.flatten(0, 1) for value in storage.distribution_params)
        metrics = {}
        weighted_kl = 0.
        total = 0
        actor = cast(HistoryGaussianStudentMLP, self._raw_actor)
        for task in range(4):
            indices = masks[:, task].nonzero(as_tuple=True)[0]
            count = indices.numel()
            if not count:
                continue
            positions = torch.linspace(0, count - 1, min(count, 1024), device=indices.device).long()
            selected = indices[positions]
            distribution = cast(GaussianTaskDistribution, actor.distribution)
            distribution.update(actor.mlp(actor.get_latent(observations[selected])))
            kl = actor.get_kl_divergence(tuple(value[selected] for value in old), actor.output_distribution_params)
            mean_kl = float(kl.mean())
            metrics[f'student_ppo_task{task}_sampled_policy_kl'] = mean_kl
            weighted_kl += count * mean_kl
            total += count
        metrics['student_ppo_sampled_policy_kl'] = weighted_kl / max(1, total)
        return metrics

    def reference_coefficient(self):
        policy_updates = max(0, self.completed_updates - self.critic_warmup_iterations)
        fraction = min(1., policy_updates / self.reference_decay_iterations)
        return self.reference_start_coef + fraction * (self.reference_floor_coef - self.reference_start_coef)

    def _reference_samples(self):
        # At most reference_batch_size frames PER TASK; equal task weighting
        # prevents common locomotion tasks from dominating the retention step.
        observations = cast(TensorDict, self.storage.observations.flatten(0, 1))
        raw = torch.cat([observations[group] for group in self._raw_actor.obs_groups], dim=-1)
        tasks = raw[:, -56 + 39].round().long()
        groups = {}
        for task in self.reference_tasks:
            indices = (tasks == task).nonzero(as_tuple=True)[0]
            if indices.numel():
                chosen = indices[torch.randperm(indices.numel(), device=indices.device)[:self.reference_batch_size]]
                groups[task] = observations[chosen].detach().clone()
        return groups

    def _reference_update(self, groups):
        coefficient = self.reference_coefficient()
        metrics = {'student_ppo_reference_coefficient': float(coefficient)}
        losses = []
        for task, observations in groups.items():
            with torch.no_grad():
                reference = self.reference_actor
                reference_distribution = cast(GaussianTaskDistribution, reference.distribution)
                reference_distribution.update(reference.mlp(reference.get_latent(observations)))
                mean_ref, std_ref = reference_distribution.mean, reference_distribution.std
            actor = cast(HistoryGaussianStudentMLP, self._raw_actor)
            distribution = cast(GaussianTaskDistribution, actor.distribution)
            distribution.update(actor.mlp(actor.get_latent(observations)))
            mean, std = distribution.mean, distribution.std
            # Mirrored transitions have three independent Gaussian controls;
            # counting their duplicated six joints would double the reference KL.
            if task in (1, 3):
                mean_ref, std_ref = mean_ref[..., :3], std_ref[..., :3]
                mean, std = mean[..., :3], std[..., :3]
            kl = (torch.log(std / std_ref) + (std_ref.square() + (mean_ref - mean).square())
                  / (2 * std.square()) - .5).sum(-1).mean()
            losses.append(kl)
            metrics[f'student_ppo_reference_task{task}_kl'] = float(kl.detach())
        if losses and coefficient > 0:
            objective = coefficient * torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward()
            # Zero gradients still let Adam momentum move inactive heads. Use
            # None for unreferenced heads and critics during this actor-only step.
            router = cast(HistoryTaskRoutedNetwork, self._raw_actor.mlp)
            for task, head in enumerate(router.heads):
                if task not in groups:
                    for parameter in head.parameters():
                        parameter.grad = None
            cast(HistoryGaussianStudentMLP, self._raw_actor).clear_inactive_distribution_gradients(set(groups))
            for parameter in self.critic.parameters():
                parameter.grad = None
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            # A plain gradient step keeps reference correction independent of
            # PPO Adam momentum and preserves the coefficient's actual scale.
            # No optimizer state is created or changed by this correction.
            with torch.no_grad():
                for parameter in self.actor.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-self.reference_learning_rate)
            self.reference_updates += 1
            metrics['student_ppo_reference_loss'] = float(objective.detach())
        else:
            metrics['student_ppo_reference_loss'] = 0.
        metrics['student_ppo_reference_updates'] = float(self.reference_updates)
        return metrics

    def update(self):
        warmup = self.completed_updates < self.critic_warmup_iterations
        groups = {} if warmup else self._reference_samples()
        parameters = [(parameter, parameter.requires_grad) for parameter in self.actor.parameters()]
        try:
            if warmup:
                for parameter, _ in parameters:
                    parameter.requires_grad_(False)
                    parameter.grad = None
            losses = super().update()
        finally:
            for parameter, trainable in parameters:
                parameter.requires_grad_(trainable)
        losses.update(self._rollout_metrics)
        losses.update(self._ppo_kl_metrics())
        if not warmup:
            losses.update(self._reference_update(groups))
        else:
            losses.update({'student_ppo_reference_coefficient': self.reference_coefficient(),
                           'student_ppo_reference_loss': 0.,
                           'student_ppo_reference_updates': float(self.reference_updates)})
        self.completed_updates += 1
        return losses | {'student_ppo_critic_warmup': float(warmup),
                         'student_ppo_completed_updates': float(self.completed_updates)}

    def _contract(self):
        contract = {'version': 2, 'critic_warmup_iterations': self.critic_warmup_iterations,
                'actor_normalization': 'frozen_supervised_v1', 'objective': 'ppo_student_reference',
                'reference_tasks': list(self.reference_tasks),
                'reference_start_coef': self.reference_start_coef,
                'reference_floor_coef': self.reference_floor_coef,
                'reference_decay_iterations': self.reference_decay_iterations,
                'reference_batch_size_per_task': self.reference_batch_size,
                'reference_learning_rate': self.reference_learning_rate,
                'reference_optimizer': 'sgd_without_momentum'}
        if self.task_balanced_advantages:
            contract['policy_task_weighting'] = 'inverse_rollout_frequency_v1'
        return contract

    def save(self):
        return super().save() | {'student_ppo': {
            'contract': self._contract(), 'completed_updates': self.completed_updates,
            'reference_actor_state_dict': self.reference_actor.state_dict(),
            'reference_updates': self.reference_updates,
        }}

    def validate_resume(self, saved):
        progress = saved.get('student_ppo')
        if (not isinstance(progress, dict) or progress.get('contract') != self._contract()
                or type(progress.get('completed_updates')) is not int or progress['completed_updates'] < 0):
            raise ValueError('student PPO resume requires a matching PPO phase and warmup contract')
        updates = progress.get('reference_updates')
        reference = progress.get('reference_actor_state_dict')
        current = self.reference_actor.state_dict()
        if (type(updates) is not int or not 0 <= updates <= max(0, progress['completed_updates'] - self.critic_warmup_iterations)
                or not isinstance(reference, dict) or current.keys() != reference.keys()
                or any(not isinstance(reference[key], torch.Tensor) or current[key].shape != reference[key].shape
                       for key in current)):
            raise ValueError('student PPO resume requires valid frozen reference state and progress')

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if not _actor_only(load_cfg):
            self.validate_resume(loaded_dict)
        result = super().load(loaded_dict, load_cfg, strict)
        if not _actor_only(load_cfg):
            self.completed_updates = loaded_dict['student_ppo']['completed_updates']
            self.reference_actor.load_state_dict(loaded_dict['student_ppo']['reference_actor_state_dict'], strict=True)
            self.reference_updates = loaded_dict['student_ppo']['reference_updates']
        return result


class StudentPPOValidationStop(RuntimeError):
    """A completed, saved validation detected persistent capability regression."""


class UniversalStudentPPORunner(MjlabOnPolicyRunner):
    """Separate cold actor transfer, full PPO resume, and actor-only playback."""

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        if train_cfg.get('console_mode', 'compact') not in ('compact', 'full'):
            raise ValueError('console_mode must be compact or full')
        interval = train_cfg.get('validation_interval_iterations', 0)
        if type(interval) is not int or interval < 0:
            raise ValueError('validation_interval_iterations must be a nonnegative integer')
        count = train_cfg.get('validation_num_envs', 80)
        seconds = train_cfg.get('validation_seconds', 8.)
        if type(count) is not int or count < 80 or count % 80:
            raise ValueError('validation_num_envs must be a positive multiple of 80')
        if not math.isfinite(seconds) or seconds < 2:
            raise ValueError('validation_seconds must be finite and at least two')
        selection = train_cfg.get('validation_selection_enabled', False)
        tolerance = train_cfg.get('validation_regression_tolerance', .25)
        patience = train_cfg.get('validation_regression_patience', 3)
        if type(selection) is not bool or (selection and not interval):
            raise ValueError('validation selection requires a boolean flag and periodic validation')
        if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
            raise ValueError('validation_regression_tolerance must be finite and in [0, 1]')
        if type(patience) is not int or patience < 1:
            raise ValueError('validation_regression_patience must be a positive integer')
        training = log_dir is not None
        source = train_cfg.get('initial_student_checkpoint')
        if training and not train_cfg.get('resume', False) and not source:
            raise ValueError('fresh student PPO training requires initial_student_checkpoint')
        super().__init__(env, train_cfg, log_dir, device)
        self.student_source = None
        self._student_validator = None
        self._student_log_dir = log_dir
        self._pending_baseline_metrics = {}
        self._validation_selector = None
        self._validation_best_checkpoint = None
        if training:
            self._publish_progress()
        if training and not train_cfg.get('resume', False):
            self.initialize_student(source, train_cfg.get('initial_student_sha256'))
        if training:
            algorithm = cast(StudentPPO, self.alg)
            physical = self.env.unwrapped
            metrics = StudentTaskMetrics(env.num_envs, device, physical.step_dt)
            process_step, update = algorithm.process_env_step, algorithm.update

            def process_with_metrics(obs, rewards, dones, extras):
                previous = algorithm.storage.observations['teacher'][algorithm.storage.step]
                reasons = {name: physical.termination_manager.get_term(name)
                           for name in physical.cfg.terminations}
                metrics.observe(previous, rewards, dones, reasons)
                return process_step(obs, rewards, dones, extras)

            def update_with_metrics():
                losses = update()
                self._publish_progress()
                losses.update(metrics.flush())
                if self._pending_baseline_metrics:
                    losses.update(self._pending_baseline_metrics)
                    self._pending_baseline_metrics = {}
                policy_updates = max(0, algorithm.completed_updates - algorithm.critic_warmup_iterations)
                if interval and policy_updates > 0 and policy_updates % interval == 0:
                    losses.update(self._run_validation())
                    if getattr(self, '_validation_stop_reason', None):
                        raise StudentPPOValidationStop(self._validation_stop_reason)
                return losses

            algorithm.process_env_step = process_with_metrics
            algorithm.update = update_with_metrics
        if training and train_cfg.get('console_mode', 'compact') == 'compact':
            from .training_console import install_compact_logging

            install_compact_logging(self.logger)

    def _run_validation(self, *, baseline=False):
        from .student_ppo_evaluation import StudentPPOEvaluator

        algorithm = cast(StudentPPO, self.alg)
        if self._student_validator is None:
            self._student_validator = StudentPPOEvaluator(
                device=self.device, num_envs=self.cfg.get('validation_num_envs', 80),
                seconds=self.cfg.get('validation_seconds', 8.),
                env_cfg=self.env.unwrapped.cfg, randomized=False,
            )
        self._student_validator.rl_iteration = max(0, algorithm.completed_updates - algorithm.critic_warmup_iterations)
        values = self._student_validator(self.alg.actor)
        decision = None
        if self.cfg.get('validation_selection_enabled', False):
            from .student_ppo_selection import StudentPPOCheckpointSelector

            if getattr(self, '_validation_selector', None) is None:
                self._validation_selector = StudentPPOCheckpointSelector(
                    values, tolerance=self.cfg.get('validation_regression_tolerance', .25),
                    patience=self.cfg.get('validation_regression_patience', 3),
                )
                decision = {'eligible': True, 'improved': True, 'should_stop': False,
                            'score': list(self._validation_selector.best_score), 'reason': 'baseline'}
            elif not baseline:
                assert self._validation_selector is not None
                decision = self._validation_selector.observe(values)
            if decision and decision['should_stop']:
                self._validation_stop_reason = decision['reason']
        label = 'baseline' if baseline else 'policy'
        report_values = {key: value if math.isfinite(value) else None for key, value in values.items()}
        print(f"Student validation {label} at update {algorithm.completed_updates}: "
              f"biped={values.get('biped_moving_success', float('nan')):.0%} "
              f"snake={values.get('snake_moving_success', float('nan')):.0%} "
              f"down/up handoff={values.get('descent_handoff_fraction', float('nan')):.0%}/"
              f"{values.get('ascent_handoff_fraction', float('nan')):.0%}")
        if self._student_log_dir is not None:
            path = Path(self._student_log_dir) / f'validation_{label}_{algorithm.completed_updates:06d}.json'
            checkpoint = None
            if not baseline or decision is not None:
                name = 'model_baseline.pt' if baseline else f'model_{algorithm.completed_updates - 1}.pt'
                checkpoint = Path(self._student_log_dir) / name
                previous_iteration = self.current_learning_iteration
                try:
                    self.current_learning_iteration = algorithm.completed_updates - 1
                    if decision and decision['improved']:
                        self._validation_best_checkpoint = str(checkpoint.resolve())
                    self.save(str(checkpoint), {'student_validation': report_values})
                    if decision and decision['improved']:
                        best = Path(self._student_log_dir) / 'model_best.pt'
                        self.save(str(best), {'student_validation': report_values})
                        (Path(self._student_log_dir) / 'validation_best.json').write_text(json.dumps({
                            'completed_updates': algorithm.completed_updates,
                            'checkpoint': str(best), 'source_checkpoint': str(checkpoint),
                            'metrics': report_values, 'selection': decision,
                        }, indent=2, allow_nan=False))
                finally:
                    self.current_learning_iteration = previous_iteration
            path.write_text(json.dumps({'completed_updates': algorithm.completed_updates,
                                        'policy_updates': self._student_validator.rl_iteration,
                                        'checkpoint': str(checkpoint) if checkpoint else None,
                                        'baseline': baseline, 'metrics': report_values,
                                        'selection': decision}, indent=2, allow_nan=False))
        prefix = 'validation_baseline_' if baseline else 'validation_'
        return {prefix + key: value for key, value in values.items() if math.isfinite(value)}

    def _publish_progress(self):
        algorithm = cast(StudentPPO, self.alg)
        physical = cast(Any, self.env.unwrapped)
        physical._student_pretraining = algorithm.completed_updates < algorithm.critic_warmup_iterations
        physical._student_rl_iteration = max(0, algorithm.completed_updates - algorithm.critic_warmup_iterations)

    def _schema(self):
        return {'role': 'universal_sensor_history_student', 'schema_version': 1,
                'frame_dim': 56, 'history_length': cast(GuidedStudentEnvCfg, self.env.unwrapped.cfg).student_history_length,
                'imu_reference': 'base', 'task_routing': 'software_fsm_v1',
                'task_encoding': 'one_hot_recovery_v1',
                'action_contract': 'gaussian_position6_rate6_mirror3_v1',
                'task_action_dims': (6, 3, 6, 3), 'teacher_sha256': TEACHER_SHA256}

    def _validate_student(self, saved):
        if (saved.get('infos') or {}).get('universal_student') != self._schema():
            raise ValueError('checkpoint is not a matching universal sensor-history student')

    def _phase(self):
        return {'version': 2, 'phase': 'student_ppo_reference',
                'environment_contract': 'student_ppo_v2_strict_handoff'}

    def initialize_student(self, checkpoint, expected_sha256=None):
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('initial student SHA256 mismatch')
        saved = torch.load(path, map_location=self.device, weights_only=False)
        self._validate_student(saved)
        if not isinstance(saved.get('student_dagger'), dict) or 'student_ppo' in saved:
            raise ValueError('cold PPO transfer requires a supervised DAgger student checkpoint')
        actor = self.alg._raw_actor
        # Keep configured small exploration instead of copying DAgger's unused std.
        distribution = {key: value.detach().clone() for key, value in actor.state_dict().items()
                        if key.startswith('distribution.')}
        actor.load_state_dict(saved['actor_state_dict'], strict=True)
        actor.load_state_dict(distribution, strict=False)
        cast(StudentPPO, self.alg).initialize_reference()
        self.student_source = {'path': str(path.resolve()), 'sha256': digest,
                               'iteration': saved.get('iter')}
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        self.env.reset()
        print(f"Student PPO fine-tuning: initialized Actor + frozen normalization from {path.resolve()}")
        print(f"Source iteration: {saved.get('iter')} | SHA256: {digest}")
        print('Critic and PPO optimizer are fresh; PPO update counter starts at 0. No teacher is loaded.')

    def save(self, path, infos=None):
        selector = getattr(self, '_validation_selector', None)
        selection = {'student_ppo_selection': selector.state_dict(),
                     'student_ppo_best_checkpoint': getattr(self, '_validation_best_checkpoint', None)
                     } if selector is not None else {}
        return super().save(path, (infos or {}) | {
            'universal_student': self._schema(),
            'student_ppo': self._phase() | {'source': self.student_source},
        } | selection)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self._validate_student(saved)
        if _actor_only(load_cfg):
            phase = (saved.get('infos') or {}).get('student_ppo')
            if 'student_dagger' not in saved and (not isinstance(phase, dict)
                                                 or phase.get('version') not in (1, 2)
                                                 or phase.get('phase') not in ('student_reward_only_ppo', 'student_ppo_reference')):
                raise ValueError('playback requires a DAgger or student PPO checkpoint')
            self.alg.load(saved, load_cfg, strict)
            return saved.get('infos')
        phase = (saved.get('infos') or {}).get('student_ppo')
        if not isinstance(phase, dict) or any(phase.get(k) != v for k, v in self._phase().items()):
            raise ValueError('full resume requires a student PPO checkpoint')
        algorithm = cast(StudentPPO, self.alg)
        algorithm.validate_resume(saved)
        selector = None
        if getattr(self, 'cfg', {}).get('validation_selection_enabled', False):
            from .student_ppo_selection import StudentPPOCheckpointSelector

            state = (saved.get('infos') or {}).get('student_ppo_selection')
            if state is None:
                raise ValueError('validation-protected resume requires saved selection state')
            selector = StudentPPOCheckpointSelector.from_state_dict(state)
            if (selector.tolerance != self.cfg.get('validation_regression_tolerance', .25)
                    or selector.patience != self.cfg.get('validation_regression_patience', 3)):
                raise ValueError('validation-protected resume requires matching selection settings')
        result = super().load(path, load_cfg, strict, map_location)
        self._validation_selector = selector
        self._validation_best_checkpoint = (saved.get('infos') or {}).get('student_ppo_best_checkpoint')
        log_dir = getattr(self, '_student_log_dir', None)
        if selector is not None and self._validation_best_checkpoint and log_dir is not None:
            # A resumed run can retain a historical best without ever improving it.
            # Preserve its artifact/provenance instead of silently losing model_best.
            source = Path(self._validation_best_checkpoint)
            best = Path(log_dir) / 'model_best.pt'
            if source.is_file() and source.resolve() != best.resolve():
                shutil.copyfile(source, best)
            (Path(log_dir) / 'validation_best.json').write_text(json.dumps({
                'checkpoint': str(best) if source.is_file() else str(source),
                'source_checkpoint': str(source), 'available': source.is_file(),
                'metrics': selector.best_metrics, 'resumed': True,
            }, indent=2, allow_nan=False))
        self.student_source = phase.get('source')
        if load_cfg is None or load_cfg.get('iteration', False):
            self.current_learning_iteration = algorithm.completed_updates
            self._publish_progress()
            self.env.reset()
        return result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        try:
            if self.cfg.get('validation_interval_iterations', 0):
                self._pending_baseline_metrics = self._run_validation(baseline=True)
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        except StudentPPOValidationStop as error:
            self.current_learning_iteration = cast(StudentPPO, self.alg).completed_updates - 1
            print(f'Student PPO stopped after validation: {error}. '
                  'See validation_policy_*.json and validation_best.json; the rejected checkpoint is preserved.')
        finally:
            if self._student_validator is not None:
                self._student_validator.close()
                self._student_validator = None
            writer = self.logger.writer
            for name in ('flush', 'close'):
                callback = getattr(writer, name, None)
                if callable(callback):
                    callback()
