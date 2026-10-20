"""Playback selects the saved encoding before creating simulator tensors."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.omni_teacher import OmniTeacherRunner


def checkpoint(path, version=3, dimension=56, **changes):
    metadata = {
        'role': 'privileged_omni_teacher', 'schema_version': version,
        'actor_obs_dim': dimension, 'critic_obs_dim': dimension, 'action_dim': 6,
    } | changes
    torch.save({'infos': {'privileged_omni_teacher': metadata}}, path)


@pytest.mark.parametrize('version,dimension,encoding', [(2, 59, 'one_hot'), (3, 56, 'scalar')])
def test_play_selects_encoding_from_saved_schema(tmp_path, version, dimension, encoding):
    path = tmp_path / 'model.pt'
    checkpoint(path, version, dimension)
    command = SimpleNamespace(task_observation_encoding='initial')
    cfg = SimpleNamespace(commands={'transform': command})
    OmniTeacherRunner.prepare_playback_env_cfg(cfg, path)
    assert command.task_observation_encoding == encoding


@pytest.mark.parametrize('changes', [
    {'role': 'student'}, {'schema_version': 1}, {'schema_version': True},
    {'actor_obs_dim': 59}, {'critic_obs_dim': 59}, {'action_dim': 3},
])
def test_invalid_schema_is_rejected_without_changing_config(tmp_path, changes):
    path = tmp_path / 'model.pt'
    checkpoint(path, **changes)
    command = SimpleNamespace(task_observation_encoding='scalar')
    with pytest.raises(ValueError):
        OmniTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={'transform': command}), path)
    assert command.task_observation_encoding == 'scalar'


def test_missing_metadata_is_rejected(tmp_path):
    path = tmp_path / 'model.pt'
    torch.save({'infos': {}}, path)
    with pytest.raises(ValueError, match='metadata'):
        OmniTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={}), path)


def test_play_prepares_saved_schema_before_environment_construction():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'play.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    prepare = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'prepare_playback_env_cfg')
    construct = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == 'ManagerBasedRlEnv')
    assert prepare.lineno < construct.lineno
