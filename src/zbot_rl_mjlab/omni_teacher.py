"""Pure PPO for one privileged policy covering every site-frame locomotion skill."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from .tasks.universal.observation_layout import (
    LEGACY_PRIVILEGED_OMNI_OBS_DIM,
    LEGACY_SITE_SNAKE_OBS_DIM,
    PRIVILEGED_OMNI_OBS_DIM,
    SITE_SNAKE_OBS_DIM,
    teacher_motion_fields,
    teacher_task_ids,
)


@dataclass
class OmniTeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
    console_mode: str = "compact"
    validation_interval_iterations: int = 0
    robust_validation_interval: int = 0


def _full_resume(load_cfg: dict | None) -> bool:
    return load_cfg is None or all(load_cfg.get(key, False) for key in ("actor", "critic", "optimizer", "iteration"))


@torch.no_grad()
def teacher_rollout_metrics(algorithm: Any) -> dict[str, float]:
    """Measure collected frames, independent of unequal task episode lengths.

    Observations are the unnormalized privileged schema, before PPO clears its
    storage. Aggregate on device and copy the scalar vector to CPU only once.
    These diagnostics enter TensorBoard's loss dictionary, not the console.
    """
    storage = getattr(algorithm, 'storage', None)
    if storage is None:
        return {}
    observations = storage.observations['actor']
    if observations.ndim != 3 or observations.shape[-1] not in (
        PRIVILEGED_OMNI_OBS_DIM, LEGACY_PRIVILEGED_OMNI_OBS_DIM, SITE_SNAKE_OBS_DIM, LEGACY_SITE_SNAKE_OBS_DIM,
    ):
        raise ValueError('teacher rollout metrics require actor observations shaped (steps, envs, 56, 59, 61 or 63)')
    if observations.shape[0] == 0 or observations.shape[1] == 0:
        raise ValueError('teacher rollout metrics require a nonempty collected rollout')
    frames = observations.reshape(-1, observations.shape[-1])
    tasks = teacher_task_ids(frames)
    command, actual, _ = teacher_motion_fields(frames)
    values: dict[str, torch.Tensor] = {}
    for phase, name in enumerate(('biped', 'descending', 'snake', 'ascending')):
        selected = tasks == phase
        count = selected.sum()
        values[f'rollout_{name}_frames'] = count
        values[f'rollout_{name}_fraction'] = count / frames.shape[0]
        if phase not in (0, 2):
            continue
        moving = selected & (command.norm(dim=-1) > .01)
        active_count = moving.sum()
        values[f'rollout_{name}_active_frames'] = active_count
        denominator = active_count.clamp_min(1)
        squared_error = torch.where(moving[:, None], (actual - command).square(), 0.)
        values[f'rollout_{name}_velocity_rmse'] = (squared_error.sum() / (3 * denominator)).sqrt()
        for axis, label in enumerate(('vx', 'vy', 'wz')):
            values[f'rollout_{name}_{label}_rmse'] = (squared_error[:, axis].sum() / denominator).sqrt()
            directed = moving & (command[:, axis].abs() > 1e-6)
            response = torch.where(directed, actual[:, axis] * command[:, axis].sign(), 0.)
            values[f'rollout_{name}_signed_{label}'] = response.sum() / directed.sum().clamp_min(1)
        planar_speed = command[:, :2].norm(dim=-1)
        translating = moving & (planar_speed > 1e-6)
        progress = (actual[:, :2] * command[:, :2]).sum(-1) / planar_speed.clamp_min(1e-6)
        values[f'rollout_{name}_planar_progress'] = torch.where(translating, progress, 0.).sum() / translating.sum().clamp_min(1)
    return dict(zip(values, torch.stack(tuple(values.values())).cpu().tolist(), strict=True))


class OmniTeacherRunner(MjlabOnPolicyRunner):
    """A standard PPO runner with progress, independent evaluation and resume guards."""

    allowed_algorithm_classes: tuple[str, ...] = ('PPO', 'zbot_rl_mjlab.balanced_ppo:TaskBalancedPPO')

    def __init__(self, env: RslRlVecEnvWrapper, train_cfg, log_dir=None, device="cpu"):
        if train_cfg.get("console_mode", "compact") not in ("compact", "full"):
            raise ValueError("console_mode must be compact or full")
        for name, default in (("validation_interval_iterations", 0), ("robust_validation_interval", 0)):
            value = train_cfg.get(name, default)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if train_cfg.get("algorithm", {}).get("class_name") not in self.allowed_algorithm_classes:
            raise ValueError("privileged omni teacher requires standard PPO")
        super().__init__(env, train_cfg, log_dir, device)
        self.completed_ppo_iterations = 0
        self._evaluator = self._robust_evaluator = None
        self._training = log_dir is not None
        self._validation_interval = train_cfg.get("validation_interval_iterations", 0)
        self._robust_interval = train_cfg.get("robust_validation_interval", 0)
        if self._training:
            if train_cfg.get("console_mode", "compact") == "compact":
                from .training_console import install_compact_logging

                install_compact_logging(self.logger)
            from .universal_omni_evaluation import UniversalOmniEvaluator

            source_cfg = env.unwrapped.cfg
            if self._validation_interval:
                self._evaluator = UniversalOmniEvaluator(device=device, env_cfg=source_cfg, randomized=False)
            if self._robust_interval:
                self._robust_evaluator = UniversalOmniEvaluator(device=device, env_cfg=source_cfg, randomized=True)
            update = self.alg.update

            def update_with_progress():
                rollout_metrics = teacher_rollout_metrics(self.alg)
                losses = dict(update())
                losses.update(rollout_metrics)
                self.completed_ppo_iterations += 1
                self._publish_progress()
                losses["teacher_ppo_iterations"] = float(self.completed_ppo_iterations)
                for evaluator, interval, prefix in (
                    (self._evaluator, self._validation_interval, ""),
                    (self._robust_evaluator, self._robust_interval, "robust_"),
                ):
                    if evaluator is not None and self.completed_ppo_iterations % interval == 0:
                        evaluator.rl_iteration = self.completed_ppo_iterations
                        losses.update({prefix + key: value for key, value in evaluator(self.alg.actor).items()})
                return losses

            self.alg.update = update_with_progress
            self._publish_progress()

    def _publish_progress(self):
        physical = cast(Any, self.env.unwrapped)
        physical._student_rl_iteration = self.completed_ppo_iterations
        physical._student_pretraining = False

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint: str | Path) -> None:
        """Select the saved observation encoding before allocating an environment."""
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        infos = saved.get("infos") or {}
        metadata = infos.get("privileged_omni_teacher")
        if not isinstance(metadata, dict) or metadata.get("role") != "privileged_omni_teacher":
            raise ValueError("checkpoint lacks privileged omni teacher observation metadata; select a teacher checkpoint")
        version = metadata.get("schema_version")
        dimension = metadata.get("actor_obs_dim")
        if (
            type(version) is not int or type(dimension) is not int
            or metadata.get("critic_obs_dim") != dimension
            or type(metadata.get("critic_obs_dim")) is not int
            or metadata.get("action_dim") != 6 or type(metadata.get("action_dim")) is not int
            or (version, dimension) not in ((2, 59), (3, 56))
        ):
            raise ValueError("unsupported teacher observation schema; expected schema 2 / 59 dimensions or schema 3 / 56 dimensions")
        command = env_cfg.commands.get("transform")
        if command is None or not hasattr(command, "task_observation_encoding"):
            raise ValueError("teacher playback requires a privileged transform command configuration")
        command.task_observation_encoding = "scalar" if dimension == 56 else "one_hot"

    def _schema(self):
        observations = self.env.get_observations()
        groups = self.cfg["obs_groups"]
        actor_dim = sum(observations[name].shape[-1] for name in groups["actor"])
        return {
            "role": "privileged_omni_teacher", "schema_version": 3 if actor_dim == 56 else 2,
            "actor_obs_dim": sum(observations[name].shape[-1] for name in groups["actor"]),
            "critic_obs_dim": sum(observations[name].shape[-1] for name in groups["critic"]),
            "action_dim": self.env.num_actions,
        }

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Preserve standard PPO's opt-in episode-length randomization."""
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            for evaluator in (self._evaluator, self._robust_evaluator):
                if evaluator is not None:
                    evaluator.close()
            # Upstream stops remote writers, but leaves TensorBoard's writer open.
            writer = self.logger.writer
            for method in ("flush", "close"):
                callback = getattr(writer, method, None)
                if callable(callback):
                    callback()

    def save(self, path, infos=None):
        from .tasks.universal.mdp import command_term

        metadata = self._schema() | {"completed_ppo_iterations": self.completed_ppo_iterations, "task_fsm_version": 1}
        course = command_term(self.env.unwrapped).tracking_ema.detach().cpu().tolist()
        return super().save(path, {
            **(infos or {}), "privileged_omni_teacher": metadata, "universal_course": course,
        })

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        from .tasks.universal.mdp import command_term

        full = _full_resume(load_cfg)
        metadata, course = None, None
        term = command_term(self.env.unwrapped)
        if full:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            infos = saved.get("infos") or {}
            metadata = infos.get("privileged_omni_teacher")
            expected = self._schema()
            if not isinstance(metadata, dict) or any(
                type(metadata.get(key)) is not type(value) or metadata.get(key) != value
                for key, value in expected.items()
            ):
                raise ValueError("checkpoint privileged omni teacher role/schema mismatch; use matching observation encoding or start a new run")
            count = metadata.get("completed_ppo_iterations")
            if type(count) is not int or count < 0:
                raise ValueError("checkpoint completed PPO iteration count is invalid")
            try:
                course = term.tracking_ema.new_tensor(infos["universal_course"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("checkpoint universal course is missing or invalid") from error
            if course.shape != term.tracking_ema.shape or not torch.isfinite(course).all() or not ((0 <= course) & (course <= 1)).all():
                raise ValueError("checkpoint universal course is invalid")
        previous_steps = self.env.unwrapped.common_step_counter
        previous_iteration = self.current_learning_iteration
        infos = super().load(path, load_cfg, strict, map_location)
        if not self._training and isinstance(infos, dict):
            saved_metadata = infos.get('privileged_omni_teacher') or {}
            if isinstance(saved_metadata, dict) and saved_metadata.get('task_fsm_version') != 1:
                print('[INFO] 回放旧检查点：该模型尚未采用显式任务状态机训练，更新代码不会改变已学到的权重。')
        if full:
            assert metadata is not None and course is not None
            term.tracking_ema.copy_(course)
            self.completed_ppo_iterations = metadata["completed_ppo_iterations"]
            # Upstream saves a zero-based loop index; continue after completed updates.
            self.current_learning_iteration = self.completed_ppo_iterations
        else:
            self.env.unwrapped.common_step_counter = previous_steps
            self.current_learning_iteration = previous_iteration
        if self._training:
            self._publish_progress()
        return infos
