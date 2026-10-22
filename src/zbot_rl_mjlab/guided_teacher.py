"""Privileged universal teacher guided by frozen, task-scoped archive policies."""

import hashlib
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import torch
from mjlab.rl import RslRlPpoAlgorithmCfg

from .gaussian_action_models import GaussianActionRoutedTaskNormalizedMLP
from .guided_ppo import GuidedTaskPPO, GuidedTaskPpoCfg
from .native_action_models import NativeActionRoutedTaskNormalizedMLP
from .omni_teacher import OmniTeacherRunnerCfg, _full_resume
from .task_models import (
    LocalFrameRoutedTaskNormalizedMLP,
    LocalFrameTaskNormalizedCritics,
    OneHotRoutedTaskNormalizedMLP,
    OneHotTaskNormalizedCritics,
    RoutedTaskNormalizedMLP,
    TaskNormalizedCritics,
)
from .task_teacher import RoutedTaskTeacherRunner
from .teachers import DEFAULT_CHECKPOINTS, SNAKE_CHECKPOINT, SNAKE_SITE_CHECKPOINT


@dataclass
class GuidedTeacherRunnerCfg(OmniTeacherRunnerCfg):
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=GuidedTaskPpoCfg)


def guidance_sources(checkpoints: dict[str, str] | None, *, include_omni=True, include_snake=False, include_site_snake=True) -> dict[str, dict[str, str]]:
    """Record frozen experts, including the site-frame snake archive."""
    if include_snake and include_site_snake:
        raise ValueError('choose only one snake guidance expert')
    defaults = {'transform': DEFAULT_CHECKPOINTS['transform']}
    if include_omni:
        defaults['omni'] = DEFAULT_CHECKPOINTS['omni']
    if include_snake:
        defaults['snake'] = SNAKE_CHECKPOINT
    if include_site_snake:
        defaults['snake_site'] = SNAKE_SITE_CHECKPOINT
    if checkpoints and checkpoints.keys() - defaults.keys():
        raise ValueError('guided teacher only accepts checkpoints for its enabled task experts')
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for name, value in (defaults | (checkpoints or {})).items():
        path = Path(value).expanduser()
        path = path if path.is_absolute() else root / path
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        sources[name] = {'path': str(path.resolve()), 'sha256': digest}
    return sources


class GuidedTeacherRunner(RoutedTaskTeacherRunner):
    allowed_algorithm_classes = ('zbot_rl_mjlab.guided_ppo:GuidedTaskPPO',)
    model_class_pairs = ((RoutedTaskNormalizedMLP, TaskNormalizedCritics),
                         (OneHotRoutedTaskNormalizedMLP, OneHotTaskNormalizedCritics),
                         (LocalFrameRoutedTaskNormalizedMLP, LocalFrameTaskNormalizedCritics),
                         (NativeActionRoutedTaskNormalizedMLP, LocalFrameTaskNormalizedCritics),
                         (GaussianActionRoutedTaskNormalizedMLP, LocalFrameTaskNormalizedCritics))

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        playback_encoding = getattr(getattr(env.unwrapped, 'cfg', None), '_guided_task_input_encoding', None)
        if log_dir is None and playback_encoding is not None:
            pair = self.model_class_pairs[('scalar_v1', 'one_hot_v1', 'one_hot_local_v1', 'one_hot_local_native_actions_v1', 'one_hot_local_gaussian_mirror_v1').index(playback_encoding)]
            train_cfg = train_cfg | {key: train_cfg[key] | {'class_name': f'{model.__module__}:{model.__name__}'}
                                     for key, model in zip(('actor', 'critic'), pair, strict=True)}
            if playback_encoding == 'one_hot_local_native_actions_v1':
                train_cfg['actor'] = train_cfg['actor'] | {'distribution_cfg': {'class_name': 'zbot_rl_mjlab.native_action_models:NativeTaskDistribution', 'init_std': 1., 'std_type': 'scalar', 'action_range': (-1., 1.)}}
            elif playback_encoding != 'one_hot_local_gaussian_mirror_v1':
                train_cfg['actor'] = train_cfg['actor'] | {'distribution_cfg': {'class_name': 'BetaDistribution', 'action_range': (-1., 1.)}}
        source_cfg = getattr(env.unwrapped, 'cfg', None)
        if source_cfg is not None:
            local_model = train_cfg['actor']['class_name'].endswith((':LocalFrameRoutedTaskNormalizedMLP', ':NativeActionRoutedTaskNormalizedMLP', ':GaussianActionRoutedTaskNormalizedMLP'))
            if bool(getattr(source_cfg, 'guided_local_reference', False)) != local_model:
                raise ValueError('guided observation reference must match the registered model encoding')
            native_model = train_cfg['actor']['class_name'].endswith((':NativeActionRoutedTaskNormalizedMLP', ':GaussianActionRoutedTaskNormalizedMLP'))
            if bool(getattr(source_cfg, 'native_biped_actions', False)) != native_model:
                raise ValueError('guided action interface must match the registered model distribution')
            gaussian_model = train_cfg['actor']['class_name'].endswith(':GaussianActionRoutedTaskNormalizedMLP')
            if bool(getattr(source_cfg, 'all_gaussian_actions', False)) != gaussian_model or bool(getattr(source_cfg, 'mirrored_transform_actions', False)) != gaussian_model:
                raise ValueError('guided Gaussian mirror interface must match the model')
        super().__init__(env, train_cfg, log_dir, device)
        self.guidance_sources: dict[str, Any] = {}
        algorithm = cast(GuidedTaskPPO, self.alg)
        if self._training and algorithm.guidance_enabled:
            from .guided_teachers import GuidedTeacherEnsemble

            checkpoints = algorithm.teacher_checkpoints
            flags = {'include_omni': 0 in algorithm.guidance_tasks, 'include_site_snake': 2 in algorithm.guidance_tasks}
            self.guidance_sources = guidance_sources(checkpoints, **flags)
            algorithm.teacher_provenance = self.guidance_sources
            algorithm.teachers = GuidedTeacherEnsemble(env.unwrapped, device=device, checkpoints=checkpoints, **flags)

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint: str | Path) -> None:
        GuidedTeacherRunner._prepare_playback_env_cfg(env_cfg, checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        metadata = saved['infos']['privileged_omni_teacher']
        env_cfg._guided_task_input_encoding = metadata.get('task_input_encoding', 'scalar_v1')
        env_cfg.guided_local_reference = env_cfg._guided_task_input_encoding in ('one_hot_local_v1', 'one_hot_local_native_actions_v1', 'one_hot_local_gaussian_mirror_v1')
        env_cfg.native_biped_actions = env_cfg._guided_task_input_encoding in ('one_hot_local_native_actions_v1', 'one_hot_local_gaussian_mirror_v1')
        env_cfg.all_gaussian_actions = env_cfg._guided_task_input_encoding == 'one_hot_local_gaussian_mirror_v1'
        env_cfg.mirrored_transform_actions = env_cfg.all_gaussian_actions
        if hasattr(env_cfg, 'actions') and not env_cfg.native_biped_actions:
            from .tasks.universal.guided_transitions import GuidedTransitionActionCfg

            allowed = {item.name for item in fields(GuidedTransitionActionCfg)}
            action = env_cfg.actions['joint_pos']
            env_cfg.actions['joint_pos'] = GuidedTransitionActionCfg(**{key: value for key, value in asdict(action).items() if key in allowed})
        command = env_cfg.commands['transform']
        command.aligned_teacher_sampling = metadata.get('teacher_sampling_contract', 'legacy_shared_v1') == 'native_archive_v1'
        if not command.aligned_teacher_sampling:
            command.velocity_limits = (.15, .08, .4)
            if hasattr(env_cfg, 'actions'):
                env_cfg.actions['joint_pos'].target_limit = math.pi


    @classmethod
    def _metadata_schema(cls) -> dict[str, Any]:
        return super()._metadata_schema() | {
            'role': 'privileged_omni_guided_teacher',
            'training_algorithm': 'task_guided_ppo_v1',
            'teacher_routes': ('omni', 'transform', 'snake_site_omni', 'transform'),
            'guidance_reward_recipe': 'archive_scoped_v1',
            'transition_rules': 'mirror_straighten_v1',
        }

    def _schema(self) -> dict[str, Any]:
        schema = super()._schema()
        tasks = self.cfg['algorithm'].get('guidance_tasks', (0, 1, 2, 3))
        schema['teacher_routes'] = ('omni' if 0 in tasks else 'ppo_only', 'transform',
                                    'snake_site_omni' if 2 in tasks else 'ppo_only', 'transform')
        name = self.cfg['actor']['class_name']
        gaussian = name.endswith(':GaussianActionRoutedTaskNormalizedMLP')
        native = gaussian or name.endswith(':NativeActionRoutedTaskNormalizedMLP')
        local = native or name.endswith(':LocalFrameRoutedTaskNormalizedMLP')
        one_hot = local or name.endswith(':OneHotRoutedTaskNormalizedMLP')
        schema['task_input_encoding'] = 'one_hot_local_gaussian_mirror_v1' if gaussian else 'one_hot_local_native_actions_v1' if native else 'one_hot_local_v1' if local else 'one_hot_v1' if one_hot else 'scalar_v1'
        schema['network_input_dim'] = 61 if local else 60 if one_hot else 56
        if native:
            schema['action_contract'] = 'gaussian_position6_rate6_mirror3_v1' if gaussian else 'native_gaussian_position_beta_rate_v1'
        if gaussian:
            schema['task_action_dims'] = (6, 3, 6, 3)
        command = getattr(getattr(self.env.unwrapped, 'cfg', None), 'commands', {}).get('transform')
        schema['teacher_sampling_contract'] = ('native_archive_v1' if getattr(command, 'aligned_teacher_sampling', False)
                                               else 'legacy_shared_v1')

        if local:
            schema.update(motion_reference='base_biped_foot0_snake_v1',
                          angular_velocity_frame='reference_local_v1', heading_encoding='sin_cos_v1',
                          observation_noise=False, observation_history_length=1)
        return schema

    @classmethod
    def _validate_metadata(cls, saved: dict) -> None:
        metadata = (saved.get('infos') or {}).get('privileged_omni_teacher')
        expected = cls._metadata_schema()
        routes = metadata.get('teacher_routes') if isinstance(metadata, dict) else None
        allowed = {(biped, 'transform', snake, 'transform')
                   for biped in ('omni', 'ppo_only') for snake in ('snake_forward', 'snake_site_omni', 'ppo_only')}
        legacy = routes == ('ppo_only', 'transform', 'snake_forward', 'transform') and isinstance(metadata, dict) and 'transition_rules' not in metadata
        if not isinstance(metadata, dict) or not isinstance(routes, tuple) or routes not in tuple(allowed) or any(
            type(metadata.get(key)) is not type(value) or metadata.get(key) != value
            for key, value in expected.items() if key != 'teacher_routes' and not (legacy and key == 'transition_rules')
        ):
            raise ValueError('checkpoint task teacher role/schema/algorithm mismatch')
        encoding = metadata.get('task_input_encoding', 'scalar_v1')
        dimension = metadata.get('network_input_dim', 56)
        dimensions = {'scalar_v1': 56, 'one_hot_v1': 60, 'one_hot_local_v1': 61, 'one_hot_local_native_actions_v1': 61, 'one_hot_local_gaussian_mirror_v1': 61}
        if encoding not in dimensions or type(dimension) is not int or dimension != dimensions[encoding]:
            raise ValueError('checkpoint task input encoding/schema mismatch')
        if metadata.get('teacher_sampling_contract', 'legacy_shared_v1') not in ('native_archive_v1', 'legacy_shared_v1'):
            raise ValueError('checkpoint teacher sampling contract mismatch')
        if encoding in ('one_hot_local_v1', 'one_hot_local_native_actions_v1', 'one_hot_local_gaussian_mirror_v1'):
            required = {'motion_reference': 'base_biped_foot0_snake_v1',
                        'angular_velocity_frame': 'reference_local_v1', 'heading_encoding': 'sin_cos_v1',
                        'observation_noise': False, 'observation_history_length': 1}
            if any(type(metadata.get(key)) is not type(value) or metadata.get(key) != value
                   for key, value in required.items()):
                raise ValueError('checkpoint local-reference observation contract mismatch')
        if encoding == 'one_hot_local_native_actions_v1' and metadata.get('action_contract') != 'native_gaussian_position_beta_rate_v1':
            raise ValueError('checkpoint native action contract mismatch')
        if encoding == 'one_hot_local_gaussian_mirror_v1' and (metadata.get('action_contract') != 'gaussian_position6_rate6_mirror3_v1' or metadata.get('task_action_dims') != (6, 3, 6, 3) or type(metadata.get('task_action_dims')) is not tuple):
            raise ValueError('checkpoint Gaussian mirror action contract mismatch')

    def _upgrade_legacy_guidance(self, saved: dict) -> dict:
        """Migrate only the published two-expert v1 contract; tensors stay intact."""
        self._validate_metadata(saved)
        algorithm = cast(GuidedTaskPPO, self.alg)
        if 2 in algorithm.guidance_tasks and 'snake_site' in self.guidance_sources:
            raise ValueError('legacy guidance uses a different snake expert; start a new three-teacher run')
        target = algorithm._guidance_contract()
        old = saved['guided_task_ppo']
        infos = saved.get('infos') or {}
        old_sources = infos.get('guidance_sources')
        old_settings = {key: value for key, value in target['settings'].items() if key != 'guidance_tasks'}
        if (type(old.get('version')) is not int or old['version'] != 1
                or old.get('guided_tasks') != [1, 2, 3] or old.get('task_names') != target['task_names']
                or old.get('settings') != old_settings or not isinstance(old_sources, dict)
                or set(old_sources) != {'transform', 'snake'} or old.get('provenance') != old_sources
                or any(old_sources.get(key) != value for key, value in self.guidance_sources.items() if key in old_sources)):
            raise ValueError('legacy guided resume requires unchanged original experts and guidance settings')
        for key in ('coefficients', 'agreement_ema', 'agreement_passes'):
            if not isinstance(old.get(key), list) or len(old[key]) != 4:
                raise ValueError('invalid legacy guidance progress')
        if old['coefficients'][0] != 0 or old['agreement_ema'][0] is not None or old['agreement_passes'][0] != 0:
            raise ValueError('invalid legacy unguided biped progress')
        for task in range(4):
            coefficient, ema, passes = (old[key][task] for key in ('coefficients', 'agreement_ema', 'agreement_passes'))
            if (not isinstance(coefficient, (float, int)) or not math.isfinite(coefficient)
                    or not 0 <= coefficient <= algorithm.guidance_start_coef
                    or (task > 0 and coefficient < algorithm.guidance_floor_coef)
                    or (ema is not None and (not isinstance(ema, (float, int)) or not math.isfinite(ema) or ema < 0))
                    or type(passes) is not int or passes < 0):
                raise ValueError('invalid legacy guidance progress')
        result = deepcopy(saved)
        progress = result['guided_task_ppo']
        progress.update(target)
        for task in range(4):
            if task not in algorithm.guidance_tasks:
                progress['coefficients'][task] = 0.
                progress['agreement_ema'][task] = None
                progress['agreement_passes'][task] = 0
            elif task not in old['guided_tasks']:
                progress['coefficients'][task] = algorithm.guidance_start_coef
                progress['agreement_ema'][task] = None
                progress['agreement_passes'][task] = 0
        result['infos']['privileged_omni_teacher'].update(self._metadata_schema())
        result['infos']['privileged_omni_teacher']['teacher_routes'] = self._schema()['teacher_routes']
        result['infos']['guidance_sources'] = self.guidance_sources
        result['infos']['guidance_upgrade'] = {'from_version': 1, 'to_version': 2,
                                               'previous_sources': old_sources}
        return result

    def save(self, path, infos=None):
        return super().save(path, (infos or {}) | {'guidance_sources': self.guidance_sources})

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        self._validate_metadata(saved)
        saved_encoding = saved['infos']['privileged_omni_teacher'].get('task_input_encoding', 'scalar_v1')
        if saved_encoding != self._schema()['task_input_encoding']:
            raise ValueError('task input encoding mismatch; use a matching model or start a new run')
        if _full_resume(load_cfg) and saved['infos']['privileged_omni_teacher'].get('teacher_sampling_contract', 'legacy_shared_v1') != self._schema()['teacher_sampling_contract']:
            raise ValueError('teacher sampling contract mismatch; start a new run for aligned teacher ranges')
        if _full_resume(load_cfg):
            if isinstance(saved.get('guided_task_ppo'), dict) and saved['guided_task_ppo'].get('version') == 1:
                saved = self._upgrade_legacy_guidance(saved)
                # Load a private copy. Never rewrite the user's archived checkpoint.
                with NamedTemporaryFile(prefix='zbot_guided_resume_', suffix='.pt') as handle:
                    torch.save(saved, handle)
                    handle.flush()
                    result = super().load(handle.name, load_cfg, strict, map_location)
                print('[INFO] 已恢复原模型和优化器，并迁移指导配置至v2。')
                return result
            if (saved.get('infos') or {}).get('guidance_sources') != self.guidance_sources:
                raise ValueError('guided teacher resume requires matching frozen teacher provenance')
        return super().load(path, load_cfg, strict, map_location)
