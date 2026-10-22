"""Online, task-balanced DAgger from the archived universal privileged teacher.

Reuse the RSL runner/model/storage protocol; this algorithm updates only the
student's mean policy. Teacher-controlled warmup seeds persistent demonstrations.
"""

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlPpoAlgorithmCfg
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .gaussian_action_models import GaussianActionRoutedTaskNormalizedMLP
from .guided_teacher import GuidedTeacherRunner, GuidedTeacherRunnerCfg
from .omni_teacher import _full_resume
from .student_metrics import StudentTaskMetrics
from .student_models import HistoryGaussianStudentMLP
from .tasks.universal.student_observations import GuidedStudentEnvCfg

TEACHER_CHECKPOINT = 'bests/universal_three_teacher_gaussian_mirror_2026-10-22_iter499/model_499.pt'
TEACHER_SHA256 = '1cac9ff22f290cc89965353f1231cbad1a5296303ddbf508bc660257b2902875'
TASK_NAMES = ('biped', 'descending', 'snake', 'ascending')


@dataclass
class StudentDAggerCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'zbot_rl_mjlab.student:StudentDAgger'
    teacher_checkpoint: str = TEACHER_CHECKPOINT
    teacher_sha256: str = TEACHER_SHA256
    dagger_updates: int = 0
    dagger_epochs: int = 1
    dagger_batch_size: int = 2048
    teacher_warmup_iterations: int = 20
    teacher_mix_iterations: int = 100
    dagger_anchor_fraction: float = 0.25
    dagger_anchor_max_samples_per_task: int = 4096
    dagger_max_samples_per_task: int = 4096
    dagger_replay_iterations: int = 4


@dataclass
class StudentRunnerCfg(GuidedTeacherRunnerCfg):
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=StudentDAggerCfg)


def load_universal_teacher(checkpoint: str, expected_sha256: str, device: str):
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != expected_sha256:
        raise ValueError('universal student teacher SHA256 mismatch')
    saved = torch.load(path, map_location=device, weights_only=False)
    GuidedTeacherRunner._validate_metadata(saved)
    metadata = saved['infos']['privileged_omni_teacher']
    if (metadata.get('task_input_encoding') != 'one_hot_local_gaussian_mirror_v1'
            or metadata.get('teacher_sampling_contract') != 'native_archive_v1'):
        raise ValueError('student requires the Gaussian mirror/local-reference universal teacher')
    observations = TensorDict({'teacher': torch.zeros(1, 56, device=device)}, batch_size=[1])
    # Infer archived capacity, never silently reconstruct a differently sized expert.
    state = saved['actor_state_dict']
    linear_weights = [value for key, value in state.items()
                      if key.startswith('mlp.heads.0.1.') and key.endswith('.weight')]
    hidden_dims = tuple(value.shape[0] for value in linear_weights[:-1])
    teacher = GaussianActionRoutedTaskNormalizedMLP(
        observations, {'actor': ['teacher']}, 'actor', 6,
        hidden_dims=hidden_dims, activation='elu', obs_normalization=True,
    ).to(device)
    teacher.load_state_dict(state, strict=True)
    teacher.eval().requires_grad_(False)
    return teacher, {'path': str(path.resolve()), 'sha256': digest}


class StudentDAgger(PPO):
    """Balanced DAgger with teacher warmup and a persistent demonstration anchor."""

    def __init__(self, *args, teacher_checkpoint=TEACHER_CHECKPOINT,
                 teacher_sha256=TEACHER_SHA256, dagger_updates=0, dagger_epochs=1,
                 dagger_batch_size=2048, dagger_max_samples_per_task=4096,
                 dagger_replay_iterations=4, teacher_warmup_iterations=20,
                 teacher_mix_iterations=100, dagger_anchor_fraction=0.25,
                 dagger_anchor_max_samples_per_task=4096, **kwargs):
        for name, value in (('dagger_epochs', dagger_epochs), ('dagger_batch_size', dagger_batch_size),
                            ('dagger_anchor_max_samples_per_task', dagger_anchor_max_samples_per_task),
                            ('dagger_max_samples_per_task', dagger_max_samples_per_task),
                            ('dagger_replay_iterations', dagger_replay_iterations)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in (('dagger_updates', dagger_updates),
                            ('teacher_warmup_iterations', teacher_warmup_iterations),
                            ('teacher_mix_iterations', teacher_mix_iterations)):
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not 0 <= dagger_anchor_fraction < 1:
            raise ValueError('dagger_anchor_fraction must be in [0, 1)')
        if kwargs.get('multi_gpu_cfg') is not None or kwargs.get('rnd_cfg') or kwargs.get('symmetry_cfg'):
            raise ValueError('student DAgger currently supports single-device training without RND/symmetry')
        super().__init__(*args, **kwargs)
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_sha256 = teacher_sha256
        self.dagger_updates = dagger_updates
        self.dagger_epochs = dagger_epochs
        self.teacher_warmup_iterations = teacher_warmup_iterations
        self.teacher_mix_iterations = teacher_mix_iterations
        self.dagger_anchor_fraction = dagger_anchor_fraction
        self.dagger_anchor_max_samples_per_task = dagger_anchor_max_samples_per_task
        self.anchor: TensorDict | None = None
        self._rollout_labels = []
        self._teacher_mask = None
        self._teacher_rows = 0
        self.dagger_batch_size = dagger_batch_size
        self.dagger_max_samples_per_task = dagger_max_samples_per_task
        self.dagger_replay_iterations = dagger_replay_iterations
        self.teacher = None
        self.teacher_source = {}
        self.completed_updates = 0
        self.replay = deque(maxlen=dagger_replay_iterations)
        self.critic.requires_grad_(False)
        # Gaussian std is retained for checkpoint/action compatibility, not learned
        # from mean labels. Collection mixes deterministic student/teacher means.
        assert self.actor.distribution is not None
        self.actor.distribution.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            (parameter for parameter in self.actor.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
        )

    def initialize_teacher(self):
        self.teacher, self.teacher_source = load_universal_teacher(
            self.teacher_checkpoint, self.teacher_sha256, self.device,
        )

    def act(self, obs):
        if self.storage.step >= self.storage.num_transitions_per_env:
            raise RuntimeError('student rollout buffer overflow')
        # TensorDict assignment copies before env.step mutates/reset observations.
        self.storage.observations[self.storage.step] = obs
        actor = cast(HistoryGaussianStudentMLP, self.actor)
        latent = actor.get_latent(obs)
        packed = actor.mlp(latent)
        assert self.actor.distribution is not None
        self.actor.distribution.update(packed)
        action = self.actor.distribution.mean.detach()
        if self.storage.step == 0:
            self._teacher_rows = 0
            self._rollout_labels = []
            self._teacher_mask = torch.rand(len(action), device=self.device) < self.teacher_beta()
        labels = None
        if self.teacher_beta() > 0:
            if self.teacher is None:
                raise RuntimeError('student training requires its frozen universal teacher')
            with torch.no_grad():
                labels = self.teacher(TensorDict({'teacher': obs['teacher']}, batch_size=[len(action)]))
            assert self._teacher_mask is not None
            action = torch.where(self._teacher_mask[:, None], labels, action)
            self._teacher_rows += self._teacher_mask.sum()
        self._rollout_labels.append(labels)
        return action

    def teacher_beta(self):
        if self.completed_updates < self.teacher_warmup_iterations:
            return 1.0
        if not self.teacher_mix_iterations:
            return 0.0
        return max(0.0, 1.0 - (self.completed_updates - self.teacher_warmup_iterations)
                   / self.teacher_mix_iterations)

    def process_env_step(self, obs, rewards, dones, extras):
        self.storage.step += 1
        self.actor.reset(dones)

    def compute_returns(self, obs):
        # Pure supervised distillation: rewards are logged but never optimized.
        pass

    @torch.no_grad()
    def _aggregate(self, frames):
        if self.teacher is None:
            raise RuntimeError('student training requires its frozen universal teacher')
        tasks = frames['teacher'][:, 39].round().long()
        recovery = (frames['teacher'][:, 39] - 3.25).abs() < 0.01
        selected = []
        counts = []
        for task in range(4):
            counts.append(int((tasks == task).sum()))
            indices = torch.where((tasks == task) & ~recovery)[0]
            selected.append(indices[torch.randperm(len(indices), device=self.device)
                                    [:self.dagger_max_samples_per_task]])
        chosen = torch.cat(selected)
        if not len(chosen):
            self.replay.append(TensorDict({
                'actor': frames['actor'][:0].clone(),
                'labels': frames['teacher'][:0, :6].clone(),
                'tasks': tasks[:0].clone(),
            }, batch_size=[0]))
            return counts, int(recovery.sum())
        if self._rollout_labels and all(label is not None for label in self._rollout_labels):
            labels = torch.cat(self._rollout_labels)[chosen]
        else:
            raw = frames['teacher'][chosen]
            labels = torch.cat([
                self.teacher(TensorDict({'teacher': chunk}, batch_size=[len(chunk)]))
                for chunk in raw.split(self.dagger_batch_size)
            ])
        # Only rate actions saturate physically. Position labels retain native units.
        labels = torch.where((tasks[chosen] != 0)[:, None], labels.clamp(-1, 1), labels)
        batch = TensorDict({'actor': frames['actor'][chosen].clone(),
                            'labels': labels.clone(), 'tasks': tasks[chosen].clone()},
                           batch_size=[len(chosen)])
        self.replay.append(batch)
        # Warmup is fully teacher-controlled; freeze its bounded anchor thereafter.
        if self.completed_updates < self.teacher_warmup_iterations:
            pool = batch if self.anchor is None else cast(TensorDict, torch.cat((self.anchor, batch)))
            keep = []
            for task in range(4):
                indices = torch.where(pool['tasks'] == task)[0]
                keep.append(indices[torch.randperm(len(indices), device=self.device)
                                    [:self.dagger_anchor_max_samples_per_task]])
            self.anchor = pool[torch.cat(keep)].clone()
        return counts, int(recovery.sum())

    def update(self):
        actor = cast(HistoryGaussianStudentMLP, self.actor)
        frames = self.storage.observations[:self.storage.step].flatten(0, 1)
        if not len(frames):
            raise RuntimeError('student DAgger cannot update an empty rollout')
        self.actor.update_normalization(frames)
        counts, recovery_count = self._aggregate(frames)
        losses = torch.zeros(4, device=self.device)
        steps = torch.zeros_like(losses)
        supervised_rows = 0
        optimizer_steps = 0
        beta = self.teacher_beta()
        if any(len(batch) for batch in self.replay):
            samples = cast(TensorDict, torch.cat(tuple(self.replay), dim=0))
            groups = [torch.where(samples['tasks'] == task)[0] for task in range(4)]
            present = [task for task, group in enumerate(groups) if len(group)]
            anchor_groups = [torch.where(self.anchor['tasks'] == task)[0]
                             if self.anchor is not None else None for task in range(4)]
            size = max(1, self.dagger_batch_size // len(present))
            epoch_steps = max((len(group) + size - 1) // size for group in groups)
            # Each replay row is visited every epoch. Smaller task groups cycle to
            # keep equal task loss weight, independently of episode length.
            epochs = 1 if self.dagger_updates else self.dagger_epochs
            for _ in range(epochs):
                orders = [group[torch.randperm(len(group), device=self.device)] for group in groups]
                for step in range(self.dagger_updates or epoch_steps):
                    terms = []
                    for task in present:
                        order = orders[task]
                        if self.dagger_updates:
                            indices = order[torch.randperm(len(order), device=self.device)[:size]]
                        else:
                            start = (step * size) % len(order)
                            indices = order[start:min(start + size, len(order))]
                        prediction = self.actor(samples[indices])
                        width = 3 if task in (1, 3) else 6
                        error = (prediction[:, :width] - samples['labels'][indices, :width]).square().mean()
                        supervised_rows += len(indices)
                        anchor_group = anchor_groups[task]
                        if self.anchor is not None and anchor_group is not None and len(anchor_group) and self.dagger_anchor_fraction:
                            anchor_size = max(1, round(len(indices) * self.dagger_anchor_fraction
                                                      / (1 - self.dagger_anchor_fraction)))
                            chosen = anchor_group[torch.randint(len(anchor_group), (anchor_size,), device=self.device)]
                            anchor_error = (self.actor(self.anchor[chosen])[:, :width]
                                            - self.anchor['labels'][chosen, :width]).square().mean()
                            error = ((1 - self.dagger_anchor_fraction) * error
                                     + self.dagger_anchor_fraction * anchor_error)
                            supervised_rows += len(chosen)
                        terms.append(error)
                        losses[task] += error.detach()
                        steps[task] += 1
                    loss = torch.stack(terms).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite student imitation loss')
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for task, head in enumerate(actor.mlp.heads):
                        if task not in present:
                            for parameter in head.parameters():
                                parameter.grad = None
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    optimizer_steps += 1
            replay_size = len(samples)
        else:
            self.optimizer.zero_grad(set_to_none=True)
            replay_size = 0
        self.completed_updates += 1
        self.storage.clear()
        self._rollout_labels = []
        metrics = {'student_updates': float(self.completed_updates),
                   'dagger_replay_samples': float(replay_size),
                   'dagger_anchor_samples': float(len(self.anchor)) if self.anchor is not None else 0.,
                   'dagger_optimizer_steps': float(optimizer_steps),
                   'dagger_supervised_rows': float(supervised_rows),
                   'dagger_teacher_beta': beta,
                   'dagger_teacher_fraction': float(self._teacher_rows / len(frames)),
                   'rollout_recovery_frames': float(recovery_count)}
        values = (losses / steps.clamp_min(1)).cpu().tolist()
        for task, name in enumerate(TASK_NAMES):
            metrics[f'dagger_{name}_mse'] = values[task]
            metrics[f'rollout_{name}_frames'] = float(counts[task])
            metrics[f'rollout_{name}_fraction'] = counts[task] / len(frames)
        return metrics

    def contract(self):
        return {'algorithm': 'universal_teacher_dagger_v2',
                'teacher_sha256': self.teacher_sha256,
                'dagger_updates': self.dagger_updates, 'dagger_epochs': self.dagger_epochs,
                'dagger_batch_size': self.dagger_batch_size,
                'teacher_warmup_iterations': self.teacher_warmup_iterations,
                'teacher_mix_iterations': self.teacher_mix_iterations,
                'dagger_anchor_fraction': self.dagger_anchor_fraction,
                'dagger_anchor_max_samples_per_task': self.dagger_anchor_max_samples_per_task,
                'dagger_max_samples_per_task': self.dagger_max_samples_per_task,
                'dagger_replay_iterations': self.dagger_replay_iterations}

    def save(self):
        return super().save() | {'student_dagger': {
            'contract': self.contract(), 'completed_updates': self.completed_updates,
            'replay': [batch.cpu() for batch in self.replay],
            'anchor': self.anchor.cpu() if self.anchor is not None else None,
        }}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        full = _full_resume(load_cfg)
        progress = loaded_dict.get('student_dagger')
        if full and (not isinstance(progress, dict) or progress.get('contract') != self.contract()):
            raise ValueError('student DAgger resume requires matching teacher and training contract')
        result = super().load(loaded_dict, load_cfg, strict)
        if full:
            assert isinstance(progress, dict)
            self.completed_updates = progress['completed_updates']
            anchor = progress.get('anchor')
            self.anchor = anchor.to(self.device) if anchor is not None else None
            self.replay = deque((batch.to(self.device) for batch in progress['replay']),
                                maxlen=self.dagger_replay_iterations)
        return result


class UniversalStudentRunner(MjlabOnPolicyRunner):
    """Keep teacher loading out of playback and reject teacher/student mixups."""

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        if log_dir is not None:
            algorithm = cast(StudentDAgger, self.alg)
            algorithm.initialize_teacher()
            physical = self.env.unwrapped
            metrics = StudentTaskMetrics(env.num_envs, device, physical.step_dt)
            process_step, update = algorithm.process_env_step, algorithm.update

            def process_with_metrics(obs, rewards, dones, extras):
                # Preserve the pre-action identity, including a terminal step
                # whose returned observations already belong to a reset robot.
                previous = algorithm.storage.observations['teacher'][algorithm.storage.step]
                reasons = {name: physical.termination_manager.get_term(name)
                           for name in physical.cfg.terminations}
                metrics.observe(previous, rewards, dones, reasons)
                return process_step(obs, rewards, dones, extras)

            def update_with_metrics():
                losses = update()
                losses.update(metrics.flush())
                return losses

            algorithm.process_env_step = process_with_metrics
            algorithm.update = update_with_metrics
            if train_cfg.get('console_mode', 'compact') == 'compact':
                from .training_console import install_compact_logging

                install_compact_logging(self.logger)
            print(f"Frozen universal teacher: {algorithm.teacher_source['path']}")
            print('Student: base IMU + encoders, 5-frame history; task-balanced online DAgger (no PPO).')

    def _schema(self):
        cfg = cast(GuidedStudentEnvCfg, self.env.unwrapped.cfg)
        return {'role': 'universal_sensor_history_student', 'schema_version': 1,
                'frame_dim': 56, 'history_length': cfg.student_history_length,
                'imu_reference': 'base', 'task_routing': 'software_fsm_v1',
                'task_encoding': 'one_hot_recovery_v1',
                'action_contract': 'gaussian_position6_rate6_mirror3_v1',
                'task_action_dims': (6, 3, 6, 3),
                'teacher_sha256': cast(StudentDAgger, self.alg).teacher_sha256}

    def save(self, path, infos=None):
        return super().save(path, (infos or {}) | {
            'universal_student': self._schema(), 'teacher_source': cast(StudentDAgger, self.alg).teacher_source,
        })

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if (saved.get('infos') or {}).get('universal_student') != self._schema():
            raise ValueError('checkpoint is not a matching universal sensor-history student')
        result = super().load(path, load_cfg, strict, map_location)
        if _full_resume(load_cfg):
            # Upstream stores the just-finished iteration, not the next one.
            self.current_learning_iteration = cast(StudentDAgger, self.alg).completed_updates
            # MJLab restores the global counter, but not physical state or task
            # clocks. Reinitialize the latter at that counter to avoid phantom
            # transformation timeouts on the first resumed environment step.
            self.env.reset()
        return result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            writer = self.logger.writer
            for method in ('flush', 'close'):
                callback = getattr(writer, method, None)
                if callable(callback):
                    callback()
