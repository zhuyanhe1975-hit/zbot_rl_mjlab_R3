"""The disturbed student is an isolated pure-DAgger sibling of experiment 22.a."""

import copy
import hashlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from test_task_contracts import canonical, task_contract

import zbot_rl_mjlab
from zbot_rl_mjlab.experiments.student import UniversalOmniStudent
from zbot_rl_mjlab.experiments.student_disturbed import UniversalOmniStudentDisturbed
from zbot_rl_mjlab.experiments.universal import StudentDisturbance
from zbot_rl_mjlab.student import (
    TEACHER_CHECKPOINT,
    TEACHER_SHA256,
    UniversalStudentRunner,
)
from zbot_rl_mjlab.tasks.common import PartialBaseImpulse
from zbot_rl_mjlab.tasks.universal.deployment import StudentBaseImpulse

ROOT = Path(__file__).resolve().parents[1]
TASK = UniversalOmniStudentDisturbed.task_id


@pytest.mark.parametrize('play', [False, True])
def test_disturbed_student_changes_only_the_existing_external_impulse_event(play):
    original = load_env_cfg(UniversalOmniStudent.task_id, play=play)
    disturbed = load_env_cfg(TASK, play=play)
    event = disturbed.events.pop('base_disturbance')
    original.events.pop('base_disturbance')
    # Includes observations, controls, rewards, physical gates and reset events.
    assert canonical(disturbed) == canonical(original)
    assert event.func is StudentBaseImpulse
    assert event.mode == 'step'
    expected = {
        'force_range': (-5., 5.), 'torque_range': (-.5, .5),
        'duration_s': (.08, .20), 'cooldown_s': (.8, 2.),
        'disturbed_fraction': 1. if play else .25, 'pretraining_scale': 1., 'ramp_iterations': 150,
    }
    for name, value in expected.items():
        assert event.params[name] == value
    assert event.params['asset_cfg'].name == 'robot'
    assert event.params['asset_cfg'].body_names == ('base',)
    addons = UniversalOmniStudentDisturbed().collected_addons()
    assert sum(type(addon) is StudentDisturbance for addon in addons) == 1


def test_disturbed_registration_retains_pure_supervision_and_original_teacher_source():
    assert TASK == UniversalOmniStudent.task_id + '-Disturbed'
    assert load_runner_cls(TASK) is UniversalStudentRunner
    original = asdict(load_rl_cfg(UniversalOmniStudent.task_id))
    disturbed = asdict(load_rl_cfg(TASK))
    assert disturbed['max_iterations'] == 150
    assert disturbed['experiment_name'] == 'zbot_universal_snake_leg_omni_student_disturbed'
    for name in ('algorithm', 'actor', 'critic', 'obs_groups'):
        assert disturbed[name] == original[name]
    algorithm = disturbed['algorithm']
    assert algorithm['class_name'] == 'zbot_rl_mjlab.student:StudentDAgger'
    assert algorithm['teacher_warmup_iterations'] == 20
    assert algorithm['teacher_mix_iterations'] == 100
    assert 150 - algorithm['teacher_warmup_iterations'] - algorithm['teacher_mix_iterations'] == 30
    assert algorithm['teacher_checkpoint'] == TEACHER_CHECKPOINT
    assert algorithm['teacher_sha256'] == TEACHER_SHA256
    assert hashlib.sha256((ROOT / TEACHER_CHECKPOINT).read_bytes()).hexdigest() == TEACHER_SHA256
    assert not any(key.startswith('initial_student_') for key in disturbed)
    assert not disturbed['resume']


def test_building_disturbed_student_preserves_every_existing_experiment():
    others = [cls for cls in zbot_rl_mjlab.EXPERIMENTS if cls.task_id != TASK]
    before = {cls.task_id: task_contract(cls) for cls in others}
    experiment = UniversalOmniStudentDisturbed()
    experiment.build_env()
    experiment.build_env(play=True)
    experiment.build_ppo()
    for cls in others:
        assert task_contract(cls) == before[cls.task_id]


def test_disturbance_addon_parameters_remain_editable_via_its_own_card(monkeypatch):
    experiment = UniversalOmniStudentDisturbed()
    card = copy.deepcopy(experiment.configuration_card())
    assert card is not None
    # The existing addon identity overrides its inherited sibling, not a second impulse.
    card['addons']['StudentDisturbance'].update(
        force_range=[-3., 4.], torque_range=[-.2, .3],
        disturbed_fraction=.4, pretraining_scale=.8, ramp_iterations=80,
    )
    card['play']['env']['events']['base_disturbance']['params']['disturbed_fraction'] = .6
    monkeypatch.setattr(experiment, 'configuration_card', lambda: card)
    for play in (False, True):
        events = experiment.build_env(play=play).events
        params = events['base_disturbance'].params
        assert params['force_range'] == (-3., 4.)
        assert params['torque_range'] == (-.2, .3)
        assert params['disturbed_fraction'] == (.6 if play else .4)
        assert params['pretraining_scale'] == .8
        assert params['ramp_iterations'] == 80
    assert load_env_cfg(TASK).events['base_disturbance'].params['force_range'] == (-5., 5.)


@pytest.mark.parametrize('pretraining', [None, True, False])
def test_default_dagger_pretraining_receives_nonzero_full_strength_disturbance(monkeypatch, pretraining):
    seen = {}
    monkeypatch.setattr(PartialBaseImpulse, '__call__', lambda self, env, ids, *args, **kwargs: seen.update(kwargs))
    cfg = load_env_cfg(TASK)
    env = SimpleNamespace(cfg=cfg)
    if pretraining is not None:
        env._student_pretraining = pretraining
    impulse = object.__new__(StudentBaseImpulse)
    impulse(env, None, **cfg.events['base_disturbance'].params)
    assert seen['force_range'] == (-5., 5.)
    assert seen['torque_range'] == (-.5, .5)
    assert seen['disturbed_fraction'] == .25


def test_quiet_environments_still_receive_zero_wrench(monkeypatch):
    from zbot_rl_mjlab.tasks import common

    seen = []
    monkeypatch.setattr(common.envs_mdp.apply_body_impulse, '__call__', lambda *args, **kwargs: None)
    impulse = object.__new__(StudentBaseImpulse)
    impulse._disturbed_mask = torch.tensor([True, False, False, False])
    impulse._num_bodies = 1
    impulse._device = 'cpu'
    impulse._body_ids = [0]
    monkeypatch.setattr(impulse, '_asset', SimpleNamespace(
        write_external_wrench_to_sim=lambda force, torque, **kwargs: seen.append((force, torque, kwargs)),
    ), raising=False)
    cfg = load_env_cfg(TASK)
    impulse(SimpleNamespace(cfg=cfg), None, **cfg.events['base_disturbance'].params)
    assert len(seen) == 1
    force, torque, kwargs = seen[0]
    assert not force.any() and not torque.any()
    assert kwargs['env_ids'].tolist() == [1, 2, 3]
    assert impulse._disturbed_mask.tolist() == [True, False, False, False]
