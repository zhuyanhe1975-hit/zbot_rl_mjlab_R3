"""Student experiment keeps teacher truth separate from sensor history."""

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from test_student_observations import environment
from test_task_contracts import task_contract

from zbot_rl_mjlab.experiments.omni_teacher import SnakeLegOmniGuidedTeacher
from zbot_rl_mjlab.experiments.student import UniversalOmniStudent
from zbot_rl_mjlab.student import (
    TEACHER_CHECKPOINT,
    TEACHER_SHA256,
    StudentRunnerCfg,
    UniversalStudentRunner,
)
from zbot_rl_mjlab.tasks.universal.guided_observations import (
    local_reference_privileged_state,
)
from zbot_rl_mjlab.tasks.universal.guided_transitions import GuidedOmniCommandCfg
from zbot_rl_mjlab.tasks.universal.native_actions import NativeGuidedTransitionActionCfg
from zbot_rl_mjlab.tasks.universal.student_observations import (
    GuidedStudentEnvCfg,
    student_sensor_state,
)

ROOT = Path(__file__).resolve().parents[1]
TASK = UniversalOmniStudent.task_id


@pytest.mark.parametrize('play', [False, True])
def test_sensor_history_and_clean_teacher_are_separate_groups(play):
    cfg = load_env_cfg(TASK, play=play)
    assert isinstance(cfg, GuidedStudentEnvCfg)
    assert list(cfg.observations) == ['actor', 'teacher']
    actor, teacher = cfg.observations['actor'], cfg.observations['teacher']
    assert actor.terms['sensors'].func is student_sensor_state
    assert teacher.terms['privileged'].func is local_reference_privileged_state
    history = actor.history_length
    assert history == cfg.student_history_length == 5
    assert actor.flatten_history_dim
    assert not teacher.history_length
    assert not actor.enable_corruption and not teacher.enable_corruption
    assert actor.terms['sensors'].noise is None
    assert teacher.terms['privileged'].noise is None
    assert cfg.student_observation_noise is (not play)
    env, _, _, _ = environment()
    assert history is not None
    assert student_sensor_state(env).shape[-1] * history == 280


@pytest.mark.parametrize('play', [False, True])
def test_inherits_native_controls_sampling_and_reference_contract(play):
    cfg = cast(GuidedStudentEnvCfg, load_env_cfg(TASK, play=play))
    baseline = load_env_cfg(SnakeLegOmniGuidedTeacher.task_id, play=play)
    assert cfg.native_biped_actions and cfg.mirrored_transform_actions and cfg.all_gaussian_actions
    assert cfg.guided_local_reference and cfg.omni_site_frame
    assert cfg.motion_site == 'foot_0_sole'
    assert isinstance(cfg.actions['joint_pos'], NativeGuidedTransitionActionCfg)
    assert cfg.actions == baseline.actions
    assert cfg.actions['joint_pos'].native_biped_scale == .25
    assert isinstance(cfg.commands['transform'], GuidedOmniCommandCfg)
    assert cfg.commands == baseline.commands
    assert cfg.commands['transform'].training_task_fractions == (.25, .25, .25, .25)
    assert cfg.play_transition_sync is play
    assert cfg.terminations == baseline.terminations


def test_single_frozen_archived_source_replaces_three_expert_guidance():
    cfg = cast(StudentRunnerCfg, load_rl_cfg(TASK))
    assert load_runner_cls(TASK) is UniversalStudentRunner
    assert cfg.obs_groups == {'actor': ('actor',), 'critic': ('teacher',)}
    assert cfg.actor.class_name == 'zbot_rl_mjlab.student_models:HistoryGaussianStudentMLP'
    assert cfg.actor.distribution_cfg is not None
    assert cfg.actor.distribution_cfg['class_name'].endswith(':GaussianTaskDistribution')
    assert cfg.actor.distribution_cfg['learn_std'] is False
    algorithm = asdict(cfg.algorithm)
    assert algorithm['class_name'] == 'zbot_rl_mjlab.student:StudentDAgger'
    assert not any(key.startswith('guidance_') for key in algorithm)
    assert 'teacher_checkpoints' not in algorithm
    assert algorithm['teacher_checkpoint'] == TEACHER_CHECKPOINT
    assert algorithm['teacher_sha256'] == TEACHER_SHA256
    assert hashlib.sha256((ROOT / TEACHER_CHECKPOINT).read_bytes()).hexdigest() == TEACHER_SHA256
    card = json.loads((ROOT / 'src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student.json').read_text())
    assert card['agent']['algorithm']['teacher_checkpoint'] == TEACHER_CHECKPOINT
    assert card['agent']['algorithm']['teacher_sha256'] == TEACHER_SHA256


def test_building_student_does_not_mutate_guided_teacher_recipe():
    before = task_contract(SnakeLegOmniGuidedTeacher)
    UniversalOmniStudent().build_env()
    UniversalOmniStudent().build_env(play=True)
    UniversalOmniStudent().build_ppo()
    assert task_contract(SnakeLegOmniGuidedTeacher) == before


def test_v2_recipe_uses_full_replay_epoch_teacher_handoff_and_demonstration_anchor():
    expected = {
        'dagger_updates': 0,
        'dagger_epochs': 1,
        'dagger_batch_size': 2048,
        'dagger_max_samples_per_task': 4096,
        'dagger_replay_iterations': 4,
        'teacher_warmup_iterations': 20,
        'teacher_mix_iterations': 100,
        'dagger_anchor_fraction': .25,
        'dagger_anchor_max_samples_per_task': 4096,
        'learning_rate': .0003,
    }
    cfg = cast(StudentRunnerCfg, load_rl_cfg(TASK))
    actual = asdict(cfg.algorithm)
    card = json.loads((ROOT / 'src/zbot_rl_mjlab/experiments/cards/universal_snake_leg_omni_student.json').read_text())
    for name, value in expected.items():
        assert actual[name] == value
        assert card['agent']['algorithm'][name] == value
    assert cfg.max_iterations == card['agent']['max_iterations'] == 500


@pytest.mark.parametrize('filename,entrypoint', [('train.sh', 'train.py'), ('run.sh', 'play.py')])
def test_r3_scripts_are_frozen_with_student_commands_preserved(filename, entrypoint):
    text = (ROOT / filename).read_text()
    active = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
    # Archived student commands stay frozen; the new reconfiguration task is active.
    active = '\n'.join(line for line in active.splitlines() if not re.match(
        rf'^\s*python\S*\s+{re.escape(entrypoint)} Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-(?:Basic|PBT)(?:\s|$)', line))
    assert not re.search(r'^\s*(?:exec\s+)?python\S*\s', active, re.MULTILINE)
    for task in (TASK, TASK + '-Disturbed'):
        assert re.search(rf'^#\s*python {re.escape(entrypoint)} {re.escape(task)}\s*\\', text, re.MULTILINE)
    assert '"$@"' in text
    if filename == 'train.sh':
        assert '--log-root logs/rsl_rl_universal_student_disturbed' in text
        assert '--agent.max-iterations 500' in text
        assert '--agent.resume False' in text
    else:
        for archive in ('universal_student_clean_2026-10-22_iter150/model_150.pt',
                        'universal_student_disturbed_2026-10-22_iter499/model_499.pt'):
            assert f'--checkpoint-file bests/{archive}' in text
