"""Guided privileged teacher role, source provenance, and playback isolation."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from zbot_rl_mjlab.guided_teacher import (
    GuidedTeacherRunner,
    guided_teacher_ppo_cfg,
)
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunner, omni_teacher_ppo_cfg
from zbot_rl_mjlab.task_ppo import TASK_NAMES


@pytest.fixture
def runner_setup(monkeypatch):
    from zbot_rl_mjlab.tasks.universal import mdp

    term = SimpleNamespace(tracking_ema=torch.zeros(2))
    physical = SimpleNamespace(common_step_counter=7)
    env = SimpleNamespace(
        unwrapped=physical, num_actions=6,
        get_observations=lambda: {"actor": torch.zeros(2, 56)},
    )
    monkeypatch.setattr(mdp, "command_term", lambda env: term)

    def base_init(self, env, cfg, log_dir, device):
        self.env, self.cfg = env, cfg
        self.alg = SimpleNamespace(update=Mock(return_value={}), actor=object())
        self.logger = SimpleNamespace(writer=None)
        self.current_learning_iteration = 0

    monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", base_init)
    runner = GuidedTeacherRunner(cast(RslRlVecEnvWrapper, env), asdict(guided_teacher_ppo_cfg()))
    return runner, term


def save_checkpoint(path, runner, **changes):
    saved = {"infos": {
        "privileged_omni_teacher": runner._schema() | {"completed_ppo_iterations": 12} | changes,
        "universal_course": [.2, .6], "guidance_sources": {},
    }}
    torch.save(saved, path)
    return saved


def test_guidance_config_keeps_successful_task_ppo_hyperparameters():
    baseline = asdict(omni_teacher_ppo_cfg())
    cfg = asdict(guided_teacher_ppo_cfg())
    assert cfg["class_name"] == "zbot_rl_mjlab.guided_teacher:GuidedTeacherRunner"
    assert cfg["algorithm"]["class_name"] == "zbot_rl_mjlab.guided_ppo:GuidedTaskPPO"
    assert cfg["critic"]["class_name"] == "zbot_rl_mjlab.task_models:TaskNormalizedCritics"
    assert cfg["actor"]["class_name"] == "zbot_rl_mjlab.task_models:RoutedTaskNormalizedMLP"
    for key, value in baseline["algorithm"].items():
        if key != "class_name":
            assert cfg["algorithm"][key] == value
    assert cfg["algorithm"]["guidance_enabled"]
    assert "warmup_iterations" not in cfg["algorithm"]


def test_schema_and_playback_select_scalar_encoding(runner_setup, tmp_path):
    runner, _ = runner_setup
    schema = runner._schema()
    assert schema["role"] == "privileged_omni_guided_teacher"
    assert schema["task_names"] == TASK_NAMES
    path = tmp_path / "grouped.pt"
    save_checkpoint(path, runner)
    command = SimpleNamespace(task_observation_encoding="one_hot")
    GuidedTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={"transform": command}), path)
    assert command.task_observation_encoding == "scalar"
    with pytest.raises(ValueError):
        OmniTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={"transform": command}), path)


@pytest.mark.parametrize("change", [
    {"role": "privileged_omni_teacher"}, {"training_algorithm": "PPO"},
    {"task_names": tuple(reversed(TASK_NAMES))}, {"task_names": list(TASK_NAMES)},
    {"schema_version": True}, {"actor_obs_dim": 59}, {"critic_obs_dim": 59}, {"action_dim": 3},
])
@pytest.mark.parametrize("load_cfg", [None, {"actor": True}])
def test_mismatched_checkpoint_rejected_before_any_weight_mutation(
    monkeypatch, runner_setup, tmp_path, change, load_cfg,
):
    runner, term = runner_setup
    path = tmp_path / "invalid.pt"
    save_checkpoint(path, runner, **change)
    load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    with pytest.raises(ValueError, match="role/schema/algorithm"):
        runner.load(path, load_cfg)
    load.assert_not_called()
    torch.testing.assert_close(term.tracking_ema, torch.zeros(2))
    assert runner.current_learning_iteration == runner.completed_ppo_iterations == 0
    command = SimpleNamespace(task_observation_encoding="untouched")
    with pytest.raises(ValueError):
        GuidedTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={"transform": command}), path)
    assert command.task_observation_encoding == "untouched"


@pytest.mark.parametrize("count", [-1, True])
def test_invalid_resume_progress_reuses_parent_guard(monkeypatch, runner_setup, tmp_path, count):
    runner, _ = runner_setup
    path = tmp_path / "invalid_progress.pt"
    save_checkpoint(path, runner, completed_ppo_iterations=count)
    load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    with pytest.raises(ValueError, match="iteration count"):
        runner.load(path)
    load.assert_not_called()


def test_full_resume_restores_parent_course_and_completed_updates(monkeypatch, runner_setup, tmp_path):
    runner, term = runner_setup
    path = tmp_path / "valid.pt"
    saved = save_checkpoint(path, runner)
    load = Mock(return_value=saved["infos"])
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    runner.load(path)
    load.assert_called_once()
    assert runner.current_learning_iteration == runner.completed_ppo_iterations == 12
    torch.testing.assert_close(term.tracking_ema, torch.tensor([.2, .6]))


def test_legacy_observation_dimension_rejected(runner_setup):
    runner, _ = runner_setup
    runner.env.get_observations = lambda: {"actor": torch.zeros(2, 59)}
    with pytest.raises(ValueError, match="56-dimensional"):
        runner._schema()


def test_resume_teacher_sources_mismatch_rejected_before_parent_load(monkeypatch, runner_setup, tmp_path):
    runner, _ = runner_setup
    path = tmp_path / "different_sources.pt"
    saved = save_checkpoint(path, runner)
    saved["infos"]["guidance_sources"] = {"snake": {"sha256": "changed"}}
    torch.save(saved, path)
    load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", load)
    with pytest.raises(ValueError, match="provenance"):
        runner.load(path)
    load.assert_not_called()


def test_teacher_provenance_hashes_exact_archives_and_rejects_old_walking():
    from zbot_rl_mjlab.guided_teacher import guidance_sources
    sources = guidance_sources(None)
    assert set(sources) == {"transform", "omni"}
    assert all(len(value["sha256"]) == 64 for value in sources.values())
    with pytest.raises(ValueError, match="only accepts"):
        guidance_sources({"walking": "missing.pt"})


def legacy_guided_checkpoint(runner):
    from test_guided_ppo import algorithm

    runner.alg = algorithm(guidance_tasks=(0, 1, 3))
    transform = {'path': 'transform.pt', 'sha256': 'a' * 64}
    runner.guidance_sources = {'transform': transform, 'omni': {'path': 'omni.pt', 'sha256': 'b' * 64}}
    runner.alg.teacher_provenance = runner.guidance_sources
    saved = deepcopy(runner.alg.save())
    saved['infos'] = {
        'privileged_omni_teacher': runner._schema() | {
            'teacher_routes': ('ppo_only', 'transform', 'snake_forward', 'transform'),
            'completed_ppo_iterations': 17,
        },
        'guidance_sources': {'transform': deepcopy(transform), 'snake': {'path': 'snake.pt', 'sha256': 'c' * 64}},
        'universal_course': [.2, .6],
    }
    progress = saved['guided_task_ppo']
    saved['infos']['privileged_omni_teacher'].pop('transition_rules')
    progress.update(version=1, guided_tasks=[1, 2, 3], iteration=17,
                    provenance=saved['infos']['guidance_sources'],
                    coefficients=[0., .7, .4, .8], agreement_ema=[None, .01, .02, .03],
                    agreement_passes=[0, 7, 8, 9])
    progress['settings'].pop('guidance_tasks')
    return saved


def assert_same_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same_tree(a, b)
    else:
        assert left == right


def test_legacy_resume_changes_guidance_only_and_preserves_learned_model(runner_setup):
    runner, _ = runner_setup
    saved = legacy_guided_checkpoint(runner)
    before = deepcopy(saved)
    upgraded = runner._upgrade_legacy_guidance(saved)
    assert upgraded['guided_task_ppo']['version'] == 2
    assert upgraded['guided_task_ppo']['guided_tasks'] == [0, 1, 3]
    assert upgraded['guided_task_ppo']['coefficients'] == [1., .7, 0., .8]
    assert upgraded['guided_task_ppo']['agreement_ema'] == [None, .01, None, .03]
    assert upgraded['guided_task_ppo']['agreement_passes'] == [0, 7, 0, 9]
    assert upgraded['guided_task_ppo']['iteration'] == 17
    assert upgraded['infos']['privileged_omni_teacher']['completed_ppo_iterations'] == 17
    assert set(upgraded['infos']['guidance_sources']) == {'omni', 'transform'}
    for key in saved:
        if key not in ('guided_task_ppo', 'infos'):
            assert_same_tree(upgraded[key], saved[key])
    assert_same_tree(saved, before)


@pytest.mark.parametrize('damage', ['settings', 'teacher', 'progress'])
def test_legacy_upgrade_rejects_unrelated_contract_changes(runner_setup, damage):
    runner, _ = runner_setup
    saved = legacy_guided_checkpoint(runner)
    if damage == 'settings':
        saved['guided_task_ppo']['settings']['guidance_updates'] += 1
    elif damage == 'teacher':
        saved['infos']['guidance_sources']['transform']['sha256'] = 'changed'
    else:
        saved['guided_task_ppo']['coefficients'][2] = float('nan')
    with pytest.raises(ValueError, match='legacy'):
        runner._upgrade_legacy_guidance(saved)


def test_old_two_teacher_checkpoint_still_prepares_for_playback(runner_setup, tmp_path):
    runner, _ = runner_setup
    path = tmp_path / 'old_two_teacher.pt'
    torch.save(legacy_guided_checkpoint(runner), path)
    command = SimpleNamespace(task_observation_encoding='one_hot')
    GuidedTeacherRunner.prepare_playback_env_cfg(SimpleNamespace(commands={'transform': command}), path)
    assert command.task_observation_encoding == 'scalar'
