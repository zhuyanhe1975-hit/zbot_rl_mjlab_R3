"""Privileged universal teacher guided by frozen, task-scoped archive policies."""

import hashlib
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import torch
from mjlab.rl import RslRlPpoAlgorithmCfg

from .guided_ppo import GuidedTaskPPO, GuidedTaskPpoCfg
from .omni_teacher import OmniTeacherRunnerCfg, _full_resume
from .task_teacher import RoutedTaskTeacherRunner
from .teachers import DEFAULT_CHECKPOINTS, SNAKE_CHECKPOINT


@dataclass
class GuidedTeacherRunnerCfg(OmniTeacherRunnerCfg):
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=GuidedTaskPpoCfg)


def guidance_sources(checkpoints: dict[str, str] | None, *, include_omni=True, include_snake=False) -> dict[str, dict[str, str]]:
    """Record enabled experts; the forward-only snake archive is opt-in."""
    defaults = {'transform': DEFAULT_CHECKPOINTS['transform']}
    if include_omni:
        defaults['omni'] = DEFAULT_CHECKPOINTS['omni']
    if include_snake:
        defaults['snake'] = SNAKE_CHECKPOINT
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

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        self.guidance_sources: dict[str, Any] = {}
        algorithm = cast(GuidedTaskPPO, self.alg)
        if self._training and algorithm.guidance_enabled:
            from .guided_teachers import GuidedTeacherEnsemble

            checkpoints = algorithm.teacher_checkpoints
            flags = {'include_omni': 0 in algorithm.guidance_tasks, 'include_snake': 2 in algorithm.guidance_tasks}
            self.guidance_sources = guidance_sources(checkpoints, **flags)
            algorithm.teacher_provenance = self.guidance_sources
            algorithm.teachers = GuidedTeacherEnsemble(env.unwrapped, device=device, checkpoints=checkpoints, **flags)

    @staticmethod
    def prepare_playback_env_cfg(env_cfg, checkpoint: str | Path) -> None:
        GuidedTeacherRunner._prepare_playback_env_cfg(env_cfg, checkpoint)

    @classmethod
    def _metadata_schema(cls) -> dict[str, Any]:
        return super()._metadata_schema() | {
            'role': 'privileged_omni_guided_teacher',
            'training_algorithm': 'task_guided_ppo_v1',
            'teacher_routes': ('omni', 'transform', 'ppo_only', 'transform'),
            'guidance_reward_recipe': 'archive_scoped_v1',
            'transition_rules': 'mirror_straighten_v1',
        }

    def _schema(self) -> dict[str, Any]:
        schema = super()._schema()
        tasks = self.cfg['algorithm'].get('guidance_tasks', (0, 1, 3))
        schema['teacher_routes'] = ('omni' if 0 in tasks else 'ppo_only', 'transform',
                                    'snake_forward' if 2 in tasks else 'ppo_only', 'transform')
        return schema

    @classmethod
    def _validate_metadata(cls, saved: dict) -> None:
        metadata = (saved.get('infos') or {}).get('privileged_omni_teacher')
        expected = cls._metadata_schema()
        routes = metadata.get('teacher_routes') if isinstance(metadata, dict) else None
        allowed = {(biped, 'transform', snake, 'transform')
                   for biped in ('omni', 'ppo_only') for snake in ('snake_forward', 'ppo_only')}
        legacy = routes == ('ppo_only', 'transform', 'snake_forward', 'transform') and isinstance(metadata, dict) and 'transition_rules' not in metadata
        if not isinstance(metadata, dict) or not isinstance(routes, tuple) or routes not in tuple(allowed) or any(
            type(metadata.get(key)) is not type(value) or metadata.get(key) != value
            for key, value in expected.items() if key != 'teacher_routes' and not (legacy and key == 'transition_rules')
        ):
            raise ValueError('checkpoint task teacher role/schema/algorithm mismatch')

    def _upgrade_legacy_guidance(self, saved: dict) -> dict:
        """Migrate only the published two-expert v1 contract; tensors stay intact."""
        self._validate_metadata(saved)
        algorithm = cast(GuidedTaskPPO, self.alg)
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
        if _full_resume(load_cfg):
            saved = torch.load(path, map_location='cpu', weights_only=False)
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
