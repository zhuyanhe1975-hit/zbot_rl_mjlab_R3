"""On-policy task PPO followed by bounded, task-balanced teacher agreement updates.

Teachers supply labels only. Rollout actions and log probabilities always belong
 to the current privileged actor; there is no warmup or scheduled phase switch.
"""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from mjlab.rl import RslRlPpoAlgorithmCfg
from tensordict import TensorDict

from .task_models import TaskRoutedNetwork
from .task_ppo import TASK_NAMES, TaskPPO, task_mask


@dataclass
class GuidedTaskPpoCfg(RslRlPpoAlgorithmCfg):
    """Three frozen experts guide biped, snake motion, and both transforms."""

    class_name: str = 'zbot_rl_mjlab.guided_ppo:GuidedTaskPPO'
    guidance_enabled: bool = True
    guidance_tasks: tuple[int, ...] = (0, 1, 2, 3)
    teacher_checkpoints: dict[str, str] | None = None
    guidance_max_samples_per_task: int = 2048
    guidance_batch_size: int = 1024
    guidance_updates: int = 4
    guidance_start_coef: float = 1.
    guidance_floor_coef: float = .05
    guidance_agreement_mse: float = .02
    guidance_agreement_min_samples: int = 32
    guidance_agreement_passes: int = 5
    guidance_decay: float = .98
    guidance_ema_beta: float = .9


class GuidedTaskPPO(TaskPPO):
    """Preserve TaskPPO's complete scalar objectives and add actor-only guidance."""

    def __init__(
        self, actor: Any, critic: Any, storage: Any, *,
        guidance_enabled: bool = True, guidance_tasks: tuple[int, ...] = (0, 1, 2, 3),
        teacher_checkpoints: dict[str, str] | None = None,
        guidance_max_samples_per_task: int = 2048, guidance_batch_size: int = 1024,
        guidance_updates: int = 4, guidance_start_coef: float = 1.,
        guidance_floor_coef: float = .05, guidance_agreement_mse: float = .02,
        guidance_agreement_min_samples: int = 32, guidance_agreement_passes: int = 5,
        guidance_decay: float = .98, guidance_ema_beta: float = .9, **kwargs: Any,
    ) -> None:
        if type(guidance_enabled) is not bool:
            raise ValueError('guidance_enabled must be boolean')
        if (not isinstance(guidance_tasks, tuple) or not guidance_tasks
                or any(type(task) is not int or not 0 <= task < 4 for task in guidance_tasks)
                or len(set(guidance_tasks)) != len(guidance_tasks)):
            raise ValueError('guidance_tasks must contain distinct integer task IDs in 0..3')
        if teacher_checkpoints is not None and (
            not isinstance(teacher_checkpoints, dict)
            or any(key not in ('omni', 'transform', 'snake', 'snake_site') or not isinstance(path, str) or not path
                   for key, path in teacher_checkpoints.items())
        ):
            raise ValueError('teacher_checkpoints only accepts nonempty omni/transform/snake/snake_site paths')
        if guidance_enabled and guidance_start_coef <= 0:
            raise ValueError('enabled guidance requires positive start coefficient')
        settings: dict[str, Any] = {
            'guidance_enabled': guidance_enabled, 'guidance_tasks': guidance_tasks,
            'guidance_max_samples_per_task': guidance_max_samples_per_task,
            'guidance_batch_size': guidance_batch_size, 'guidance_updates': guidance_updates,
            'guidance_start_coef': guidance_start_coef, 'guidance_floor_coef': guidance_floor_coef,
            'guidance_agreement_mse': guidance_agreement_mse,
            'guidance_agreement_min_samples': guidance_agreement_min_samples,
            'guidance_agreement_passes': guidance_agreement_passes,
            'guidance_decay': guidance_decay, 'guidance_ema_beta': guidance_ema_beta,
        }
        for name in ('guidance_max_samples_per_task', 'guidance_batch_size', 'guidance_updates',
                     'guidance_agreement_min_samples', 'guidance_agreement_passes'):
            if type(settings[name]) is not int or settings[name] < 1:
                raise ValueError(f'{name} must be a positive integer')
        if guidance_batch_size < len(guidance_tasks):
            raise ValueError('guidance_batch_size must cover all configured guided tasks')
        if any(not math.isfinite(float(value)) for key, value in settings.items() if key != 'guidance_tasks'):
            raise ValueError('guidance settings must be finite')
        if not 0 <= guidance_floor_coef <= guidance_start_coef or guidance_agreement_mse <= 0:
            raise ValueError('invalid guidance coefficient or agreement threshold')
        if not 0 < guidance_decay <= 1 or not 0 <= guidance_ema_beta < 1:
            raise ValueError('invalid guidance decay or EMA beta')
        super().__init__(actor, critic, storage, **kwargs)
        self.guidance_settings = settings
        self.guidance_enabled = guidance_enabled
        self.guidance_tasks = guidance_tasks
        self.guidance_max_samples_per_task = guidance_max_samples_per_task
        self.guidance_batch_size = guidance_batch_size
        self.guidance_updates = guidance_updates
        self.guidance_start_coef = guidance_start_coef
        self.guidance_floor_coef = guidance_floor_coef
        self.guidance_agreement_mse = guidance_agreement_mse
        self.guidance_agreement_min_samples = guidance_agreement_min_samples
        self.guidance_agreement_passes = guidance_agreement_passes
        self.guidance_decay = guidance_decay
        self.guidance_ema_beta = guidance_ema_beta
        self._guidance_generator = torch.Generator(device=self.device)
        self._guidance_generator.manual_seed((torch.initial_seed() + 7919) % (2 ** 63 - 1))
        self.teachers: Any = None
        self.teacher_checkpoints = teacher_checkpoints
        self.teacher_provenance: dict[str, Any] = {}
        self.guidance_coefficients = [guidance_start_coef if task in guidance_tasks else 0. for task in range(4)]
        self.guidance_agreement_ema: list[float | None] = [None] * 4
        self.guidance_passes = [0] * 4
        self.guidance_iteration = 0
        self._guidance_samples: dict[int, list[tuple[TensorDict, torch.Tensor, torch.Tensor]]] = {
            task: [] for task in self.guidance_tasks
        }
        self._guidance_counts = [0] * 4

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.guidance_enabled and self.teachers is not None:
            with torch.no_grad():
                targets = self.teachers.query()
                self._cache_guidance(obs, targets.actions, targets.weights)
        return super().act(obs)

    def _cache_guidance(self, obs: TensorDict, actions: torch.Tensor, weights: torch.Tensor) -> None:
        n = obs.batch_size[0]
        if actions.shape != (n, 6) or weights.shape != (n,):
            raise ValueError('teacher targets require actions[N,6] and weights[N]')
        raw = obs['actor']
        tasks = task_mask(raw).argmax(-1)
        valid = torch.isfinite(actions).all(-1) & torch.isfinite(weights) & (weights > 0)
        bounded = (actions.abs() <= 1).all(-1)
        valid &= bounded | ((tasks == 0) & bool(getattr(self._raw_actor, 'native_biped_actions', False))) | bool(getattr(self._raw_actor, 'all_gaussian_actions', False))
        if getattr(self._raw_actor, 'mirrored_transform_actions', False):
            from .tasks.transformation.actions import expand_mirror, project_mirror

            mirrored = expand_mirror(project_mirror(actions))
            valid &= ~((tasks == 1) | (tasks == 3)) | ((actions - mirrored).abs().amax(-1) < 1e-5)
        for key in list(obs.keys()):
            valid &= torch.isfinite(obs[key]).flatten(1).all(-1)
        for task in self.guidance_tasks:
            remaining = self.guidance_max_samples_per_task - self._guidance_counts[task]
            if remaining <= 0:
                continue
            candidates = torch.where(valid & (tasks == task))[0]
            # Spread the bounded budget over the rollout, rather than filling it
            # with initial poses in the first one or two simulation steps. A
            # separate generator avoids consuming the actor sampling RNG.
            quota = math.ceil(self.guidance_max_samples_per_task / self.storage.num_transitions_per_env)
            count = min(remaining, quota, candidates.numel())
            positions = torch.randperm(candidates.numel(), device=candidates.device,
                                       generator=self._guidance_generator)[:count]
            indices = candidates[positions]
            if not indices.numel():
                continue
            # The runner collects under inference_mode. Explicitly create normal
            # detached tensors so autograd may save these inputs during BC later.
            with torch.inference_mode(False):
                sample = (obs[indices].detach().clone(), actions[indices].detach().clone(),
                          weights[indices].detach().clone())
            self._guidance_samples[task].append(sample)
            self._guidance_counts[task] += indices.numel()

    def _clear_guidance(self) -> None:
        for samples in self._guidance_samples.values():
            samples.clear()
        self._guidance_counts = [0] * 4

    def _clear_inactive_actor_gradients(self, active_tasks: set[int]) -> None:
        router = getattr(self._raw_actor, 'mlp', None)
        if isinstance(router, TaskRoutedNetwork):
            # gather() produces zero gradients for unused heads. Adam still moves
            # such parameters using momentum from PPO unless grad is truly None.
            for task, head in enumerate(router.heads):
                if task not in active_tasks:
                    for parameter in head.parameters():
                        parameter.grad = None
        clear_distribution = getattr(self._raw_actor, 'clear_inactive_distribution_gradients', None)
        if callable(clear_distribution):
            clear_distribution(active_tasks)
        elif 0 not in active_tasks and getattr(self._raw_actor, 'native_biped_actions', False):
            gaussian = getattr(getattr(self._raw_actor, 'distribution', None), 'gaussian', None)
            assert isinstance(gaussian, torch.nn.Module)
            for parameter in gaussian.parameters():
                parameter.grad = None

    def _guidance_update(self) -> dict[str, float]:
        groups: dict[int, tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        metrics: dict[str, float] = {}
        for task, samples in self._guidance_samples.items():
            if not samples:
                continue
            obs = cast(TensorDict, torch.cat([sample[0] for sample in samples], dim=0))
            actions = torch.cat([sample[1] for sample in samples])
            weights = torch.cat([sample[2] for sample in samples])
            held_out = torch.arange(len(actions), device=actions.device) % 5 == 0
            groups[task] = (obs, actions, weights, held_out)
            metrics[f'guidance_{TASK_NAMES[task]}_cached'] = float(len(actions))
        active = [task for task, (_, _, _, held_out) in groups.items()
                  if self.guidance_coefficients[task] > 0 and (~held_out).any()]
        losses = []
        for _ in range(self.guidance_updates if active else 0):
            task_losses = []
            for task in active:
                obs, actions, weights, held_out = groups[task]
                train = torch.where(~held_out)[0]
                count = min(len(train), self.guidance_batch_size // len(active))
                indices = train[torch.randperm(len(train), device=train.device)[:count]]
                predicted = self.actor(obs[indices], stochastic_output=False)
                errors = (predicted - actions[indices]).square().mean(-1)
                batch_weights = weights[indices]
                mse = (errors * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
                task_losses.append(self.guidance_coefficients[task] * mse)
                key = f'guidance_{TASK_NAMES[task]}_optimized'
                metrics[key] = metrics.get(key, 0.) + count
            objective = torch.stack(task_losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward()
            self._clear_inactive_actor_gradients(set(active))
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses.append(objective.detach())
        with torch.no_grad():
            for task, (obs, actions, weights, held_out) in groups.items():
                count = int(held_out.sum())
                metrics[f'guidance_{TASK_NAMES[task]}_agreement_samples'] = float(count)
                if not count:
                    continue
                errors = (self.actor(obs[held_out], stochastic_output=False) - actions[held_out]).square().mean(-1)
                mse = float((errors * weights[held_out]).sum() / weights[held_out].sum().clamp_min(1e-8))
                metrics[f'guidance_{TASK_NAMES[task]}_agreement_mse'] = mse
                if count >= self.guidance_agreement_min_samples:
                    previous = self.guidance_agreement_ema[task]
                    ema = mse if previous is None else self.guidance_ema_beta * previous + (1 - self.guidance_ema_beta) * mse
                    self.guidance_agreement_ema[task] = ema
                    self.guidance_passes[task] = self.guidance_passes[task] + 1 if ema < self.guidance_agreement_mse else 0
                    if self.guidance_passes[task] >= self.guidance_agreement_passes:
                        self.guidance_coefficients[task] = max(
                            self.guidance_floor_coef, self.guidance_coefficients[task] * self.guidance_decay,
                        )
                else:
                    self.guidance_passes[task] = 0
        metrics['guidance'] = float(torch.stack(losses).mean()) if losses else 0.
        for task in self.guidance_tasks:
            metrics[f'guidance_{TASK_NAMES[task]}_coefficient'] = self.guidance_coefficients[task]
        return metrics

    def update(self) -> dict[str, float]:
        try:
            # Always optimize on-policy PPO, including iteration zero. Its normalizer
            # update remains the only update of observation statistics this rollout.
            losses = super().update()
            if self.guidance_enabled:
                losses.update(self._guidance_update())
            self.guidance_iteration += 1
            return losses
        finally:
            self._clear_guidance()

    def _guidance_contract(self) -> dict[str, Any]:
        return {'version': 2, 'task_names': list(TASK_NAMES), 'guided_tasks': list(self.guidance_tasks),
                'settings': dict(self.guidance_settings), 'provenance': self.teacher_provenance}

    def save(self) -> dict:
        result = super().save()
        result['guided_task_ppo'] = self._guidance_contract() | {
            'iteration': self.guidance_iteration,
            'coefficients': list(self.guidance_coefficients),
            'agreement_ema': list(self.guidance_agreement_ema),
            'agreement_passes': list(self.guidance_passes),
        }
        return result

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        actor_only = load_cfg is not None and load_cfg.get('actor', False) and not any(
            load_cfg.get(key, False) for key in ('critic', 'optimizer', 'iteration', 'rnd')
        )
        progress: Any = loaded_dict.get('guided_task_ppo')
        if not actor_only:
            contract = self._guidance_contract()
            if not isinstance(progress, dict) or any(progress.get(key) != value for key, value in contract.items()):
                raise ValueError('GuidedTaskPPO full resume requires matching guidance contract and teacher provenance')
            coefficients, ema, passes = (cast(Any, progress.get(key)) for key in ('coefficients', 'agreement_ema', 'agreement_passes'))
            if (not all(isinstance(value, list) and len(value) == 4 for value in (coefficients, ema, passes))
                    or type(progress.get('iteration')) is not int or progress['iteration'] < 0):
                raise ValueError('invalid guidance checkpoint progress')
            for task in range(4):
                if (not isinstance(coefficients[task], (float, int)) or not math.isfinite(coefficients[task])
                        or not 0 <= coefficients[task] <= self.guidance_start_coef
                        or (ema[task] is not None and (not isinstance(ema[task], (float, int))
                            or not math.isfinite(ema[task]) or ema[task] < 0))
                        or type(passes[task]) is not int or passes[task] < 0):
                    raise ValueError('invalid guidance checkpoint progress')
            if any((coefficients[task] < self.guidance_floor_coef if task in self.guidance_tasks
                    else coefficients[task] != 0) for task in range(4)):
                raise ValueError('invalid guidance checkpoint coefficients')
            # Validate model shapes before either upstream load_state_dict can
            # partially mutate parameters and then report an incompatible key.
            for key, model in (('actor_state_dict', self._raw_actor), ('critic_state_dict', self._raw_critic)):
                if load_cfg is not None and not load_cfg.get(key.split('_')[0], False):
                    continue
                current = model.state_dict()
                incoming = loaded_dict.get(key)
                if (not isinstance(incoming, dict) or current.keys() != incoming.keys()
                        or any(not isinstance(incoming[name], torch.Tensor)
                               or current[name].shape != incoming[name].shape for name in current)):
                    raise ValueError('GuidedTaskPPO model architecture does not match checkpoint')
        result = super().load(loaded_dict, load_cfg, strict)
        if not actor_only:
            self.guidance_coefficients = list(progress['coefficients'])
            self.guidance_agreement_ema = list(progress['agreement_ema'])
            self.guidance_passes = list(progress['agreement_passes'])
            self.guidance_iteration = progress['iteration']
            self._clear_guidance()
        return result
