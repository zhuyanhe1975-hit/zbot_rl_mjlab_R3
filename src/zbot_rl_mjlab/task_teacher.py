"""Task-routed value learning preserves each skill's complete scalar objective."""

from pathlib import Path
from typing import Any

import torch

from .omni_teacher import OmniTeacherRunner, OmniTeacherRunnerCfg, omni_teacher_ppo_cfg
from .task_ppo import TASK_NAMES


def task_teacher_ppo_cfg() -> OmniTeacherRunnerCfg:
    cfg = omni_teacher_ppo_cfg()
    cfg.class_name = 'zbot_rl_mjlab.task_teacher:TaskTeacherRunner'
    cfg.algorithm.class_name = 'zbot_rl_mjlab.task_ppo:TaskPPO'
    cfg.critic.class_name = 'zbot_rl_mjlab.task_ppo:IndependentTaskCritics'
    return cfg


class TaskTeacherRunner(OmniTeacherRunner):
    allowed_algorithm_classes = ('zbot_rl_mjlab.task_ppo:TaskPPO',)

    def _schema(self) -> dict[str, Any]:
        schema = super()._schema()
        if (schema['actor_obs_dim'], schema['critic_obs_dim'], schema['action_dim']) != (56, 56, 6):
            raise ValueError('task teacher requires scalar 56-dimensional observations and 6 actions')
        return schema | self._metadata_schema()

    @classmethod
    def _metadata_schema(cls) -> dict[str, Any]:
        return {
            'role': 'privileged_omni_task_teacher',
            'schema_version': 3,
            'actor_obs_dim': 56, 'critic_obs_dim': 56, 'action_dim': 6,
            'training_algorithm': 'task_ppo_v1',
            'task_names': TASK_NAMES,
        }

    @classmethod
    def _validate_metadata(cls, saved: dict) -> None:
        metadata = (saved.get('infos') or {}).get('privileged_omni_teacher')
        expected = cls._metadata_schema()
        if not isinstance(metadata, dict) or any(
            type(metadata.get(key)) is not type(value) or metadata.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError('checkpoint task teacher role/schema/algorithm mismatch')

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint: str | Path) -> None:
        TaskTeacherRunner._prepare_playback_env_cfg(env_cfg, checkpoint)

    @classmethod
    def _prepare_playback_env_cfg(cls, env_cfg, checkpoint: str | Path) -> None:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cls._validate_metadata(saved)
        command = env_cfg.commands.get('transform')
        if command is None or not hasattr(command, 'task_observation_encoding'):
            raise ValueError('task teacher playback requires a privileged transform command')
        command.task_observation_encoding = 'scalar'

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        self._validate_metadata(torch.load(path, map_location='cpu', weights_only=False))
        return super().load(path, load_cfg, strict, map_location)


def routed_task_teacher_ppo_cfg() -> OmniTeacherRunnerCfg:
    cfg = task_teacher_ppo_cfg()
    cfg.class_name = 'zbot_rl_mjlab.task_teacher:RoutedTaskTeacherRunner'
    cfg.actor.class_name = 'zbot_rl_mjlab.task_models:RoutedTaskNormalizedMLP'
    cfg.critic.class_name = 'zbot_rl_mjlab.task_models:TaskNormalizedCritics'
    return cfg


class RoutedTaskTeacherRunner(TaskTeacherRunner):
    """One task-routed actor with four branches and task-specific input statistics."""

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        expected = routed_task_teacher_ppo_cfg()
        for key, model in (('actor', expected.actor), ('critic', expected.critic)):
            if train_cfg[key]['class_name'] != model.class_name or not train_cfg[key].get('obs_normalization'):
                raise ValueError(f'routed teacher requires its registered {key} model and normalization')
        super().__init__(env, train_cfg, log_dir, device)

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint: str | Path) -> None:
        RoutedTaskTeacherRunner._prepare_playback_env_cfg(env_cfg, checkpoint)

    @classmethod
    def _metadata_schema(cls) -> dict[str, Any]:
        return super()._metadata_schema() | {
            'role': 'privileged_omni_task_routed_teacher',
            'training_algorithm': 'routed_task_ppo_v1',
            'actor_branches': TASK_NAMES,
            'actor_normalization': 'per_task_v1',
            'critic_normalization': 'per_task_v1',
        }
