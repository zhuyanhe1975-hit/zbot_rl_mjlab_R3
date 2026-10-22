"""CPU regressions for the independent privileged PPO runner."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlPpoAlgorithmCfg

from zbot_rl_mjlab.experiments.omni_teacher import omni_teacher_ppo_cfg
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunner


@pytest.fixture
def setup_runner(monkeypatch):
    from zbot_rl_mjlab import universal_omni_evaluation
    from zbot_rl_mjlab.tasks.universal import mdp

    term = SimpleNamespace(tracking_ema=torch.zeros(2))
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(cfg=object(), common_step_counter=7),
        num_actions=6, get_observations=lambda: {"actor": torch.zeros(2, 59)},
    )
    alg = SimpleNamespace(update=Mock(return_value={"value": 2.}), actor=object())
    writer = SimpleNamespace(flush=Mock(), close=Mock())
    logger = SimpleNamespace(writer=writer)

    def base_init(self, env, cfg, log_dir, device):
        self.env, self.cfg, self.alg, self.logger = env, cfg, alg, logger
        self.current_learning_iteration = 0

    evaluators = []

    def factory(**kwargs):
        evaluator = Mock(return_value={"validation_passed": 1.})
        evaluator.settings = kwargs
        evaluators.append(evaluator)
        return evaluator

    monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", base_init)
    monkeypatch.setattr(universal_omni_evaluation, "UniversalOmniEvaluator", factory)
    monkeypatch.setattr(mdp, "command_term", lambda env: term)
    cfg = asdict(omni_teacher_ppo_cfg())
    cfg.update(console_mode="full", validation_interval_iterations=2, robust_validation_interval=3)
    return env, alg, logger, evaluators, cfg, term


def test_config_has_one_privileged_group_and_plain_ppo():
    cfg = omni_teacher_ppo_cfg()
    assert type(cfg.algorithm) is RslRlPpoAlgorithmCfg
    assert cfg.algorithm.class_name == "PPO"
    assert cfg.obs_groups == {"actor": ("actor",), "critic": ("actor",)}
    assert cfg.actor.hidden_dims == cfg.critic.hidden_dims == (256, 128)
    assert cfg.actor.obs_normalization and cfg.critic.obs_normalization
    assert cfg.actor.distribution_cfg == {"class_name": "BetaDistribution", "action_range": (-1., 1.)}
    assert (cfg.max_iterations, cfg.num_steps_per_env, cfg.save_interval) == (6000, 24, 50)
    assert (cfg.validation_interval_iterations, cfg.robust_validation_interval) == (0, 0)


@pytest.mark.parametrize("changes", [
    {"console_mode": "silent"}, {"validation_interval_iterations": True},
    {"validation_interval_iterations": -1}, {"robust_validation_interval": 1.5},
    {"algorithm": {"class_name": "MultiTeacherPPO"}},
])
def test_invalid_cfg_rejected_before_base_constructor(monkeypatch, changes):
    base = Mock(side_effect=AssertionError("base must not run"))
    monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", base)
    with pytest.raises(ValueError):
        OmniTeacherRunner(cast(Any, None), asdict(omni_teacher_ppo_cfg()) | changes)
    base.assert_not_called()


def test_inference_constructs_no_evaluators_or_update_hook(setup_runner):
    env, alg, _, evaluators, cfg, _ = setup_runner
    original = alg.update
    runner = OmniTeacherRunner(env, cfg)
    assert not evaluators and alg.update is original
    assert runner.completed_ppo_iterations == 0
    assert not hasattr(env.unwrapped, "_student_rl_iteration")


def test_updates_publish_completed_ppo_progress_and_independent_evaluations(setup_runner):
    env, alg, _, evaluators, cfg, _ = setup_runner
    runner = OmniTeacherRunner(env, cfg, log_dir="run")
    assert [item.settings["randomized"] for item in evaluators] == [False, True]
    assert all(item.settings["env_cfg"] is env.unwrapped.cfg for item in evaluators)
    for count in range(1, 7):
        losses = alg.update()
        assert losses["value"] == 2.
        assert losses["teacher_ppo_iterations"] == count
        assert ("validation_passed" in losses) == (count % 2 == 0)
        assert ("robust_validation_passed" in losses) == (count % 3 == 0)
        assert env.unwrapped._student_rl_iteration == count
        assert env.unwrapped._student_pretraining is False
    assert runner.completed_ppo_iterations == 6
    assert [item.call_count for item in evaluators] == [3, 2]


def test_compact_logger_installed_only_for_training(monkeypatch, setup_runner):
    from zbot_rl_mjlab import training_console

    env, _, logger, _, cfg, _ = setup_runner
    install = Mock()
    monkeypatch.setattr(training_console, "install_compact_logging", install)
    cfg["console_mode"] = "compact"
    OmniTeacherRunner(env, cfg)
    install.assert_not_called()
    OmniTeacherRunner(env, cfg, log_dir="run")
    install.assert_called_once_with(logger)


@pytest.mark.parametrize("fails", [False, True])
def test_learn_closes_evaluation_and_writer_even_on_failure(monkeypatch, setup_runner, fails):
    env, _, logger, evaluators, cfg, _ = setup_runner
    runner = OmniTeacherRunner(env, cfg, log_dir="run")
    base_learn = Mock(side_effect=RuntimeError("failed") if fails else None)
    monkeypatch.setattr(MjlabOnPolicyRunner, "learn", base_learn)
    if fails:
        with pytest.raises(RuntimeError, match="failed"):
            runner.learn(5, True)
    else:
        runner.learn(5, True)
    base_learn.assert_called_once_with(5, True)
    for evaluator in evaluators:
        evaluator.close.assert_called_once()
    logger.writer.flush.assert_called_once()
    logger.writer.close.assert_called_once()


def _checkpoint(runner, course=None):
    return {"infos": {
        "privileged_omni_teacher": runner._schema() | {"completed_ppo_iterations": 12},
        "universal_course": [.2, .6] if course is None else course,
    }}


def test_save_records_actual_dimensions_and_course_without_multiteacher_metadata(monkeypatch, setup_runner):
    env, _, _, _, cfg, term = setup_runner
    runner = OmniTeacherRunner(env, cfg, log_dir="run")
    runner.completed_ppo_iterations = 9
    term.tracking_ema[:] = .25
    base_save = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "save", base_save)
    runner.save("model.pt", {"user": "data"})
    path, infos = base_save.call_args.args
    assert path == "model.pt" and infos["user"] == "data"
    assert infos["universal_course"] == [.25, .25]
    assert infos["privileged_omni_teacher"] == {
        "role": "privileged_omni_teacher", "schema_version": 2,
        "actor_obs_dim": 59, "critic_obs_dim": 59, "action_dim": 6,
        "completed_ppo_iterations": 9,
        "task_fsm_version": 1,
    }
    assert "multi_teacher" not in infos


@pytest.mark.parametrize("change", [
    {"role": "student"}, {"schema_version": 1}, {"schema_version": True},
    {"actor_obs_dim": 53}, {"critic_obs_dim": 53}, {"action_dim": 12},
    {"completed_ppo_iterations": -1}, {"completed_ppo_iterations": True},
])
def test_full_resume_rejects_mismatch_before_model_mutation(monkeypatch, setup_runner, tmp_path, change):
    env, _, _, _, cfg, _ = setup_runner
    runner = OmniTeacherRunner(env, cfg, log_dir="run")
    saved = _checkpoint(runner)
    saved["infos"]["privileged_omni_teacher"].update(change)
    path = tmp_path / "model.pt"
    torch.save(saved, path)
    base_load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", base_load)
    with pytest.raises(ValueError):
        runner.load(path)
    base_load.assert_not_called()


def test_default_training_creates_no_periodic_evaluation_at_iteration_100(setup_runner):
    env, algorithm, _, evaluators, _, _ = setup_runner
    original_update = algorithm.update
    cfg = asdict(omni_teacher_ppo_cfg())
    cfg['console_mode'] = 'full'
    runner = OmniTeacherRunner(env, cfg, log_dir='run')
    losses = {}
    for _ in range(100):
        losses = runner.alg.update()
    assert original_update.call_count == 100
    assert runner.completed_ppo_iterations == 100
    assert not evaluators
    assert runner._evaluator is runner._robust_evaluator is None
    assert set(losses) == {'value', 'teacher_ppo_iterations'}


@pytest.mark.parametrize("infos", [{}, {"universal_course": [0., 0.]}])
def test_full_resume_rejects_missing_role(monkeypatch, setup_runner, tmp_path, infos):
    env, _, _, _, cfg, _ = setup_runner
    runner = OmniTeacherRunner(env, cfg)
    path = tmp_path / "model.pt"
    torch.save({"infos": infos}, path)
    base_load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", base_load)
    with pytest.raises(ValueError, match="role/schema"):
        runner.load(path)
    base_load.assert_not_called()


@pytest.mark.parametrize("course", [[0.], [float("nan"), 0.], [2., 0.], ["bad", 0.]])
def test_full_resume_validates_course_before_mutation(monkeypatch, setup_runner, tmp_path, course):
    env, _, _, _, cfg, _ = setup_runner
    runner = OmniTeacherRunner(env, cfg)
    path = tmp_path / "model.pt"
    torch.save(_checkpoint(runner, course), path)
    base_load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", base_load)
    with pytest.raises(ValueError, match="course"):
        runner.load(path)
    base_load.assert_not_called()


def test_full_resume_restores_actual_completed_updates_and_progress(monkeypatch, setup_runner, tmp_path):
    env, alg, _, _, cfg, term = setup_runner
    runner = OmniTeacherRunner(env, cfg, log_dir="run")
    path = tmp_path / "model.pt"
    saved = _checkpoint(runner)
    torch.save(saved, path)
    monkeypatch.setattr(MjlabOnPolicyRunner, "load", lambda *args: saved["infos"])
    runner.load(path)
    assert runner.current_learning_iteration == runner.completed_ppo_iterations == 12
    assert env.unwrapped._student_rl_iteration == 12
    torch.testing.assert_close(term.tracking_ema, torch.tensor([.2, .6]))
    alg.update()
    assert runner.completed_ppo_iterations == 13


def test_partial_actor_load_retains_new_course_steps_and_iteration(monkeypatch, setup_runner):
    env, _, _, _, cfg, term = setup_runner
    runner = OmniTeacherRunner(env, cfg, log_dir="run")
    runner.current_learning_iteration = 3
    runner.completed_ppo_iterations = 3

    def base_load(*args):
        runner.current_learning_iteration = 99
        env.unwrapped.common_step_counter = 1000
        return {"universal_course": [.9, .9]}

    monkeypatch.setattr(MjlabOnPolicyRunner, "load", base_load)
    runner.load("other-role.pt", {"actor": True})
    assert runner.current_learning_iteration == runner.completed_ppo_iterations == 3
    assert env.unwrapped.common_step_counter == 7
    torch.testing.assert_close(term.tracking_ema, torch.zeros(2))


def test_scalar_schema_records_version_three(setup_runner):
    env, _, _, _, cfg, _ = setup_runner
    env.get_observations = lambda: {'actor': torch.zeros(2, 56)}
    runner = OmniTeacherRunner(env, cfg)
    assert runner._schema() == {
        'role': 'privileged_omni_teacher', 'schema_version': 3,
        'actor_obs_dim': 56, 'critic_obs_dim': 56, 'action_dim': 6,
    }


@pytest.mark.parametrize('dimension', [56, 59])
def test_cross_encoding_full_resume_rejected_before_weights(monkeypatch, setup_runner, tmp_path, dimension):
    env, _, _, _, cfg, _ = setup_runner
    runner = OmniTeacherRunner(env, cfg)
    saved = _checkpoint(runner)
    other = 59 if dimension == 56 else 56
    saved['infos']['privileged_omni_teacher'].update(
        schema_version=3 if other == 56 else 2, actor_obs_dim=other, critic_obs_dim=other,
    )
    env.get_observations = lambda: {'actor': torch.zeros(2, dimension)}
    path = tmp_path / 'other.pt'
    torch.save(saved, path)
    base_load = Mock()
    monkeypatch.setattr(MjlabOnPolicyRunner, 'load', base_load)
    with pytest.raises(ValueError, match='role/schema'):
        runner.load(path)
    base_load.assert_not_called()
