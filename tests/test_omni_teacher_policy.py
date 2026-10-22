"""Exact teacher selection and immediate PPO for site-frame omni training."""

from copy import deepcopy
from dataclasses import asdict
from inspect import unwrap
from types import SimpleNamespace
from typing import cast

import pytest
import test_universal_distillation as distillation_tests
import test_universal_teachers as teacher_tests
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from zbot_rl_mjlab.distillation import MultiTeacherPPO
from zbot_rl_mjlab.experiments.universal import universal_ppo_cfg
from zbot_rl_mjlab.teachers import TeacherEnsemble


@pytest.fixture
def env():
    return unwrap(teacher_tests.env)()


@pytest.fixture
def algorithm():
    return unwrap(distillation_tests.algorithm)()


def test_transform_only_loads_only_selected_checkpoint(env, monkeypatch):
    import zbot_rl_mjlab.teachers as module

    loaded, configs = [], []
    original_load, original_cfg = torch.load, module.load_env_cfg

    def load(path, *args, **kwargs):
        loaded.append(str(path))
        return original_load(path, *args, **kwargs)

    def config(task, *args, **kwargs):
        configs.append(task)
        return original_cfg(task, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', load)
    monkeypatch.setattr(module, 'load_env_cfg', config)
    teachers = TeacherEnsemble(env, enabled_teachers=('transform',))
    assert teachers.teacher_names == ('transform',)
    assert len(loaded) == len(configs) == 1
    assert 'transform_snake' in loaded[0]
    monkeypatch.setattr(module.walking_mdp, '_forward_world', lambda *args: pytest.fail('unused walking state'))
    targets = teachers.query()
    assert targets.counts == {'transform': 2}
    assert targets.teacher_ids is not None
    assert targets.teacher_ids.tolist() == [-1, -1, -1, -1, 0, 0]


@pytest.mark.parametrize('selection', [(), ('transform', 'transform'), ('unknown',)])
def test_invalid_selection_rejected_before_loading(env, selection):
    with pytest.raises(ValueError):
        TeacherEnsemble(env, enabled_teachers=selection)


def test_disabled_checkpoint_override_rejected(env):
    with pytest.raises(ValueError, match='未启用'):
        TeacherEnsemble(env, enabled_teachers=('transform',), checkpoints={'walking': '/missing'})


def configure_immediate(alg):
    alg.enabled_teachers = ('transform',)
    alg.teacher_warmup_enabled = False
    alg.ppo_start_iteration = 0
    alg._cached_counts = [0]
    alg.teachers = cast(TeacherEnsemble, SimpleNamespace(
        teacher_names=('transform',), query=lambda: SimpleNamespace(
            actions=torch.full((8, 6), .7), weights=torch.ones(8),
            counts={'transform': 8}, teacher_ids=torch.zeros(8, dtype=torch.long))))


def test_immediate_ppo_and_single_teacher_heldout(algorithm, monkeypatch):
    alg, obs, _ = algorithm
    configure_immediate(alg)
    calls = []
    original = PPO.update

    def update(self):
        calls.append(True)
        return original(self)

    monkeypatch.setattr(PPO, 'update', update)
    distillation_tests.rollout(alg, obs)
    assert alg._sample_count == 12
    assert torch.cat([sample[4] for sample in alg._samples]).any()
    losses = alg.update()
    assert calls == [True]
    assert losses['distillation_warmup'] == 0
    assert losses['student_rl_iteration'] == 1
    assert 'teacher_walking_cached' not in losses
    saved = deepcopy(alg.save())
    assert saved['multi_teacher']['anchor'] is None
    alg.load(saved)
    assert not alg.is_warmup and alg.ppo_start_iteration == 0


def test_full_resume_rejects_changed_teacher_semantics(algorithm):
    alg, _, _ = algorithm
    saved = deepcopy(alg.save())
    configure_immediate(alg)
    with pytest.raises(ValueError, match='教师|预热'):
        alg.load(saved)


def test_constructor_starts_real_ppo_without_teacher_gate():
    cfg = asdict(universal_ppo_cfg())
    cfg.update(multi_gpu=None, num_steps_per_env=2)
    for key in ('actor', 'critic'):
        cfg[key]['hidden_dims'] = (16,)
        for option in ('cnn_cfg', 'rnn_type', 'rnn_hidden_dim', 'rnn_num_layers'):
            cfg[key].pop(option, None)
    cfg['critic'].pop('distribution_cfg')
    cfg['algorithm'].update(enabled_teachers=('transform',), teacher_warmup_enabled=False)
    obs = TensorDict({'actor': torch.ones(8, 7), 'critic': torch.ones(8, 7)}, batch_size=[8])
    env_proxy = SimpleNamespace(num_envs=8, num_actions=6)
    alg = cast(MultiTeacherPPO, MultiTeacherPPO.construct_algorithm(obs, cast(VecEnv, env_proxy), cfg, 'cpu'))
    assert alg.teacher_names == ('transform',)
    assert alg.ppo_start_iteration == 0 and not alg.is_warmup
    assert alg._cached_counts == [0]
    alg.load(deepcopy(alg.save()))
    assert alg.ppo_start_iteration == 0 and not alg.is_warmup


def test_unversioned_checkpoint_cannot_resume_into_explicit_selection(algorithm):
    alg, _, _ = algorithm
    saved = deepcopy(alg.save())
    saved['multi_teacher'].pop('teacher_names')
    alg.enabled_teachers = ('transform',)
    with pytest.raises(ValueError, match='教师选择'):
        alg.load(saved)


@pytest.mark.parametrize('selection', [(), ('transform', 'transform'), ('unknown',)])
def test_algorithm_rejects_invalid_teacher_selection(selection):
    with pytest.raises(ValueError):
        MultiTeacherPPO(enabled_teachers=selection)


def test_checkpoint_validates_warmup_mode_even_with_same_teachers(algorithm):
    alg, _, _ = algorithm
    saved = deepcopy(alg.save())
    alg.teacher_warmup_enabled = False
    with pytest.raises(ValueError, match='预热'):
        alg.load(saved)
