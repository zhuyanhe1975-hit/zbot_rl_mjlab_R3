"""综合策略的蒸馏阶段、优化与恢复契约。"""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from zbot_rl_mjlab.distillation import MultiTeacherPPO, universal_ppo_cfg
from zbot_rl_mjlab.teachers import TeacherEnsemble


@pytest.fixture
def algorithm():
    torch.manual_seed(9)
    cfg = asdict(universal_ppo_cfg())
    cfg.update(multi_gpu=None, num_steps_per_env=2)
    for key in ("actor", "critic"):
        cfg[key]["hidden_dims"] = (16,)
        for option in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            cfg[key].pop(option, None)
    cfg["critic"].pop("distribution_cfg")
    cfg["algorithm"].update(
        warmup_iterations=2, anneal_iterations=2,
        distillation_max_samples=12, distillation_batch_size=8, distillation_updates=3,
        learning_rate=0.02, schedule="fixed", num_learning_epochs=1, num_mini_batches=1,
    )
    obs = TensorDict({"actor": torch.ones(8, 7), "critic": torch.ones(8, 7)}, batch_size=[8])
    env = SimpleNamespace(num_envs=8, num_actions=6)
    alg = cast(MultiTeacherPPO, MultiTeacherPPO.construct_algorithm(obs, cast(VecEnv, env), cfg, "cpu"))
    teacher_parameter = torch.nn.Parameter(torch.full((8, 6), 0.7))
    alg.teachers = cast(TeacherEnsemble, SimpleNamespace(query=lambda: SimpleNamespace(
        actions=teacher_parameter, weights=torch.ones(8), counts={"walking": 2, "omni": 2, "transform": 4},
        teacher_ids=torch.tensor([0, 0, 1, 1, 2, 2, 2, 2]),
    )))
    return alg, obs, teacher_parameter


def rollout(alg, obs):
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.ones(8), torch.zeros(8), {})
        alg.compute_returns(obs)


def test_warmup_learns_with_frozen_teacher_and_fresh_critic(algorithm):
    alg, obs, _teacher_parameter = algorithm
    optimizer = alg.optimizer
    critic = deepcopy(alg.critic.state_dict())
    before = (alg.actor(obs, stochastic_output=False) - 0.7).square().mean().item()
    rollout(alg, obs)
    assert alg._sample_count == 12
    assert not alg._samples[0][0]["actor"].is_inference()
    metrics = alg.update()
    after = (alg.actor(obs, stochastic_output=False) - 0.7).square().mean().item()
    assert after < before
    assert alg.optimizer is optimizer
    assert _teacher_parameter.grad is None
    assert all(torch.equal(value, critic[key]) for key, value in alg.critic.state_dict().items())
    assert metrics["distillation_warmup"] == 1
    assert metrics["distillation_coverage"] == 1
    assert metrics["teacher_walking_candidates"] == 4
    assert metrics["teacher_transform_candidates"] == 8
    assert metrics["teacher_walking_cached"] == 4
    assert metrics["teacher_walking_optimized"] == metrics["teacher_transform_optimized"]
    assert alg.storage.step == 0


def test_phase_transition_calls_ppo_and_anneals(algorithm, monkeypatch):
    alg, obs, _teacher_parameter = algorithm
    from rsl_rl.algorithms import PPO

    alg.validation_interval_iterations = 1
    alg.validation_required_passes = 1
    alg.validation_min_samples = 1
    alg.validation_max_mse = 2
    alg.validation_max_std = 1
    alg.validation_callback = lambda: {"validation_passed": True}
    calls = []
    original = PPO.update

    def track(self):
        calls.append(True)
        return original(self)

    monkeypatch.setattr(PPO, "update", track)
    metrics = {}
    for _ in range(3):
        rollout(alg, obs)
        metrics = alg.update()
    assert len(calls) == 1
    assert metrics["distillation_warmup"] == 0
    assert "value" in metrics
    assert alg.imitation_coef == pytest.approx(0.51)
    alg.distillation_iteration += 1
    assert alg.imitation_coef == pytest.approx(0.02)


def test_no_valid_warmup_fails_and_ppo_remains_available(algorithm):
    alg, obs, _teacher_parameter = algorithm
    alg.teachers = SimpleNamespace(query=lambda: SimpleNamespace(
        actions=torch.zeros(8, 6), weights=torch.zeros(8), counts={},
    ))
    rollout(alg, obs)
    with pytest.raises(RuntimeError, match="没有有效教师"):
        alg.update()
    assert alg.distillation_iteration == 0
    alg.distillation_enabled = False
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["distillation"] == 0
    assert metrics["distillation_coefficient"] == 0
    assert "value" in metrics


def test_full_resume_and_actor_only_hot_start(algorithm):
    alg, obs, _teacher_parameter = algorithm
    rollout(alg, obs)
    alg.update()
    saved = deepcopy(alg.save())
    assert alg.optimizer.state
    alg.distillation_iteration = 99
    assert alg.load(saved, None, True)
    assert alg.distillation_iteration == 1
    assert alg.optimizer.state
    assert not alg.load(saved, {"actor": True, "iteration": True}, True)
    assert alg.distillation_iteration == 0
    assert not alg.load(saved, {"actor": True, "critic": True, "iteration": True}, True)
    assert alg.distillation_iteration == 0
    saved.pop("multi_teacher")
    with pytest.raises(ValueError, match="完整恢复"):
        alg.load(saved, None, True)


def test_algorithm_accepts_no_teacher_for_inference(algorithm):
    alg, obs, _teacher_parameter = algorithm
    alg.teachers = None
    with torch.inference_mode():
        action = alg.actor(obs, stochastic_output=False)
    assert action.shape == (8, 6)
    assert torch.all(action.abs() <= 1)
    assert alg.actor.mlp[-2].out_features == 12


def test_runner_inference_never_opens_missing_teachers(algorithm, monkeypatch):
    from mjlab.rl import MjlabOnPolicyRunner

    from zbot_rl_mjlab.distillation import MultiTeacherRunner

    alg, _, _ = algorithm
    alg.teacher_checkpoints = {"missing": "/does/not/exist.pt"}

    def base_init(self, env, cfg, log_dir, device):
        self.alg = alg

    monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", base_init)
    runner = MultiTeacherRunner(cast(RslRlVecEnvWrapper, None), {}, log_dir=None)
    assert cast(MultiTeacherPPO, runner.alg).teachers is alg.teachers


def test_runner_save_restore_and_hot_start_environment(algorithm, tmp_path):
    from zbot_rl_mjlab.distillation import MultiTeacherRunner

    alg, obs, _teacher_parameter = algorithm
    rollout(alg, obs)
    alg.update()
    runner = MultiTeacherRunner.__new__(MultiTeacherRunner)
    runner.alg = alg
    runner.env = cast(RslRlVecEnvWrapper, SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=123)))
    runner.cfg = {"upload_model": False}
    runner.current_learning_iteration = 10
    checkpoint = tmp_path / "student.pt"
    runner.save(str(checkpoint))
    runner.env.unwrapped.common_step_counter = 1
    runner.current_learning_iteration = 0
    runner.load(str(checkpoint))
    assert runner.env.unwrapped.common_step_counter == 123
    assert runner.current_learning_iteration == 10
    assert alg.distillation_iteration == 1
    runner.env.unwrapped.common_step_counter = 7
    runner.load(str(checkpoint), {"actor": True})
    assert runner.env.unwrapped.common_step_counter == 7
    assert alg.distillation_iteration == 0


def test_full_resume_restores_schedule_but_hot_start_keeps_new_configuration(algorithm):
    alg, _, _ = algorithm
    alg.warmup_iterations = 1
    alg.anneal_iterations = 7
    alg.start_coef = 0.8
    alg.floor_coef = 0.03
    alg.distillation_iteration = 4
    saved = deepcopy(alg.save())
    expected_coef = alg.imitation_coef
    alg.warmup_iterations = 100
    alg.anneal_iterations = 1000
    alg.start_coef = 1
    alg.floor_coef = 0.02
    alg.load(saved)
    assert alg.distillation_iteration == 4
    assert alg.warmup_iterations == 1
    assert alg.anneal_iterations == 7
    assert alg.imitation_coef == pytest.approx(expected_coef)
    alg.warmup_iterations = 25
    alg.distillation_enabled = False
    alg.load(saved, {"actor": True})
    assert alg.warmup_iterations == 25
    assert alg.distillation_iteration == 0
    assert not alg.distillation_enabled
    alg.load(saved)
    assert not alg.distillation_enabled
    saved["multi_teacher"].pop("schedule")
    alg.warmup_iterations = 17
    alg.load(saved)
    assert alg.warmup_iterations == 17


@pytest.mark.parametrize("change", ["hash", "keys"])
def test_runner_rejects_changed_teachers_before_mutating_training_state(algorithm, tmp_path, change):
    from zbot_rl_mjlab.distillation import MultiTeacherRunner

    alg, _, _ = algorithm
    runner = MultiTeacherRunner.__new__(MultiTeacherRunner)
    runner.alg = alg
    runner.env = cast(RslRlVecEnvWrapper, SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=123)))
    runner.cfg = {"upload_model": False}
    runner.current_learning_iteration = 10
    alg.teacher_provenance = {"walking": {"path": "/old.pt", "sha256": "original"}}
    checkpoint = tmp_path / "student.pt"
    runner.save(str(checkpoint))
    if change == "hash":
        alg.teacher_provenance["walking"]["sha256"] = "changed"
    else:
        alg.teacher_provenance = {"other": {"path": "/old.pt", "sha256": "original"}}
    with torch.no_grad():
        next(alg.actor.parameters()).fill_(42)
    actor_before = deepcopy(alg.actor.state_dict())
    alg.optimizer.param_groups[0]["lr"] = 0.77
    runner.env.unwrapped.common_step_counter = 5
    runner.current_learning_iteration = 0
    with pytest.raises(ValueError, match="教师检查点内容"):
        runner.load(str(checkpoint))
    assert all(torch.equal(value, actor_before[key]) for key, value in alg.actor.state_dict().items())
    assert alg.optimizer.param_groups[0]["lr"] == 0.77
    assert runner.current_learning_iteration == 0
    assert runner.env.unwrapped.common_step_counter == 5


def test_runner_accepts_relocated_teachers_and_skips_check_for_hot_start_or_pure_ppo(algorithm, tmp_path):
    from zbot_rl_mjlab.distillation import MultiTeacherRunner

    alg, _, _ = algorithm
    runner = MultiTeacherRunner.__new__(MultiTeacherRunner)
    runner.alg = alg
    runner.env = cast(RslRlVecEnvWrapper, SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=123)))
    runner.cfg = {"upload_model": False}
    runner.current_learning_iteration = 10
    alg.teacher_provenance = {"walking": {"path": "/old.pt", "sha256": "same"}}
    checkpoint = tmp_path / "student.pt"
    runner.save(str(checkpoint))
    alg.teacher_provenance["walking"]["path"] = "/new/location.pt"
    runner.load(str(checkpoint))
    assert alg.teacher_provenance["walking"]["path"] == "/new/location.pt"
    alg.teacher_provenance["walking"]["sha256"] = "different"
    runner.load(str(checkpoint), {"actor": True})
    alg.distillation_enabled = False
    runner.load(str(checkpoint))
    alg.distillation_enabled = True
    alg.teachers = None
    runner.load(str(checkpoint), {"actor": True})


def test_warmup_executes_teacher_and_deterministic_fallback(algorithm):
    alg, obs, _ = algorithm
    alg.teachers = SimpleNamespace(query=lambda: SimpleNamespace(
        actions=torch.full((8, 6), 0.7), weights=torch.tensor([1.] * 4 + [0.] * 4),
        counts={"walking": 4}, teacher_ids=torch.tensor([0] * 4 + [-1] * 4),
    ))
    expected = alg.actor(obs, stochastic_output=False).detach()
    with torch.inference_mode():
        actions = alg.act(obs)
    assert torch.allclose(actions[:4], torch.full((4, 6), 0.7))
    assert torch.allclose(actions[4:], expected[4:])


def test_warmup_cannot_end_without_independent_behavior_validation(algorithm):
    alg, obs, _ = algorithm
    alg.distillation_iteration = 1000
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["distillation_warmup"] == 1
    assert alg.is_warmup
    assert metrics["warmup_blocked_behavior"] == 1


def test_missing_teacher_blocks_even_when_behavior_passes(algorithm):
    alg, obs, _ = algorithm
    alg.warmup_iterations = 0
    alg.validation_interval_iterations = 1
    alg.validation_required_passes = 1
    alg.validation_min_samples = 1
    alg.validation_max_mse = 2
    alg.validation_max_std = 1
    alg.validation_callback = lambda: {"validation_passed": True}
    alg.teachers = SimpleNamespace(query=lambda: SimpleNamespace(
        actions=torch.zeros(8, 6), weights=torch.ones(8), counts={"walking": 8},
        teacher_ids=torch.zeros(8, dtype=torch.long),
    ))
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["warmup_blocked_omni"] == 1
    assert metrics["warmup_blocked_transform"] == 1
    assert alg.is_warmup


def test_behavior_gate_requires_consecutive_passes_and_restores(algorithm):
    alg, obs, _ = algorithm
    alg.warmup_iterations = 0
    alg.validation_interval_iterations = 1
    alg.validation_required_passes = 2
    alg.validation_min_samples = 1
    alg.validation_max_mse = 2
    alg.validation_max_std = 1
    results = iter([True, False, True, True])
    alg.validation_callback = lambda: {"validation_passed": next(results), "forward_success": 1.0}
    reset_observations = obs.clone()
    reset_observations["actor"].fill_(3)
    resets = []
    alg.phase_callback = lambda enabled: resets.append(enabled) or reset_observations
    for expected in (1, 0, 1):
        rollout(alg, obs)
        metrics = alg.update()
        assert metrics["warmup_validation_passes"] == expected
        assert alg.is_warmup
    saved = deepcopy(alg.save())
    alg.validation_passes = 0
    alg.load(saved)
    assert alg.validation_passes == 1
    rollout(alg, obs)
    alg.update()
    assert not alg.is_warmup
    assert resets == [False]
    with torch.inference_mode():
        alg.act(obs)
    assert torch.all(alg.transition.observations["actor"] == 3)
    saved = deepcopy(alg.save())
    alg.ppo_start_iteration = None
    alg.load(saved)
    assert not alg.is_warmup
    alg.load(saved, {"actor": True})
    assert alg.is_warmup
    assert alg.validation_passes == 0


def test_joint_rollout_executes_student_and_records_matching_action(algorithm):
    alg, obs, _ = algorithm
    alg.ppo_start_iteration = 0
    with torch.inference_mode():
        action = alg.act(obs)
    assert torch.equal(action, alg.transition.actions)
    assert not torch.allclose(action, torch.full_like(action, 0.7))
    assert not torch.allclose(action, alg.actor(obs, stochastic_output=False))


def test_warmup_reduces_action_noise_and_handles_boundary_teacher(algorithm):
    alg, obs, teacher = algorithm
    with torch.inference_mode():
        alg.actor(obs, stochastic_output=True)
        before = alg.actor.output_std.mean().item()
    with torch.no_grad():
        teacher.fill_(1)
    alg.distillation_updates = 15
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["teacher_transform_validation_std"] < before
    assert all(torch.isfinite(value).all() for value in alg.actor.parameters())


def test_incompatible_actor_shape_is_rejected_before_load(algorithm):
    alg, _, _ = algorithm
    saved = deepcopy(alg.save())
    saved["actor_state_dict"]["mlp.0.weight"] = torch.zeros(16, 49)
    before = deepcopy(alg.actor.state_dict())
    with pytest.raises(ValueError, match="49 维"):
        alg.load(saved)
    assert all(torch.equal(value, before[key]) for key, value in alg.actor.state_dict().items())


def test_runner_wires_independent_validation_and_preserves_course(algorithm, monkeypatch, tmp_path):
    from mjlab.rl import MjlabOnPolicyRunner

    from zbot_rl_mjlab import universal_evaluation
    from zbot_rl_mjlab.distillation import MultiTeacherRunner

    alg, obs, _ = algorithm
    term = SimpleNamespace(tracking_ema=torch.tensor([.2, .7]))
    term.set_pretraining = lambda enabled: setattr(term, 'pretraining', enabled)
    physical = SimpleNamespace(common_step_counter=123, command_manager=SimpleNamespace(get_term=lambda name: term))
    resets = []
    env = SimpleNamespace(unwrapped=physical, reset=lambda: (resets.append(True) or obs, {}))

    def base_init(self, env, cfg, log_dir, device):
        self.alg, self.env = alg, env
        self.cfg = {'upload_model': False}
        self.current_learning_iteration = 4

    evaluator = SimpleNamespace(closed=False)
    evaluator.close = lambda: setattr(evaluator, 'closed', True)
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', base_init)
    monkeypatch.setattr('zbot_rl_mjlab.teachers.TeacherEnsemble', lambda *args, **kwargs: alg.teachers)
    monkeypatch.setattr(MultiTeacherRunner, '_teacher_provenance', lambda self: {})
    monkeypatch.setattr(universal_evaluation, 'UniversalEvaluator', lambda **kwargs: evaluator)
    monkeypatch.setattr(MjlabOnPolicyRunner, 'learn', lambda self, count, random_lengths: (count, random_lengths))
    runner = MultiTeacherRunner(cast(RslRlVecEnvWrapper, env), {'console_mode': 'full'}, log_dir=str(tmp_path))
    assert term.pretraining and len(resets) == 1
    assert alg.validation_callback is not None and alg.phase_callback is not None
    assert runner.learn(2, True) == (2, False)
    assert evaluator.closed
    checkpoint = tmp_path / 'course.pt'
    runner.save(str(checkpoint))
    term.tracking_ema.zero_()
    runner.load(str(checkpoint))
    torch.testing.assert_close(term.tracking_ema, torch.tensor([.2, .7]))
    assert term.pretraining and len(resets) == 2
    alg.phase_callback(False)
    assert not term.pretraining and len(resets) == 3


def test_dagger_starts_only_after_minimum_warmup_and_stores_executed_actions(algorithm):
    alg, obs, _ = algorithm
    expected = alg.actor(obs, stochastic_output=False).detach()
    alg.distillation_iteration = alg.warmup_iterations - 1
    with torch.inference_mode():
        actions = alg.act(obs)
    assert torch.allclose(actions, torch.full_like(actions, 0.7))
    alg.distillation_iteration = alg.warmup_iterations
    with torch.inference_mode():
        actions = alg.act(obs)
    torch.testing.assert_close(actions[:2], expected[:2])
    torch.testing.assert_close(actions[2:], torch.full_like(actions[2:], 0.7))
    assert torch.equal(actions, alg.transition.actions)
    assert alg.is_warmup


@pytest.mark.parametrize("num_envs", [1, 4, 8, 12, 16, 20, 32])
def test_dagger_probe_fraction_and_ability_coverage(algorithm, num_envs):
    alg, obs, _ = algorithm
    expanded = obs[torch.arange(num_envs) % len(obs)]
    alg.teachers = SimpleNamespace(query=lambda: SimpleNamespace(
        actions=torch.full((num_envs, 6), 0.7), weights=torch.ones(num_envs),
        counts={"walking": num_envs}, teacher_ids=torch.zeros(num_envs, dtype=torch.long),
    ))
    alg.distillation_iteration = alg.warmup_iterations
    with torch.inference_mode():
        action = alg.act(expanded)
    student = (action - 0.7).abs().sum(dim=-1) > 1e-5
    assert int(student.sum()) == max(1, num_envs // 4)
    if num_envs >= 16:
        assert set((student.nonzero().flatten() % 4).tolist()) == {0, 1, 2, 3}
    assert alg._warmup_student_count / alg._warmup_action_count == pytest.approx(max(1, num_envs // 4) / num_envs)


def test_dagger_logs_fraction_and_keeps_all_live_teacher_queries(algorithm):
    alg, obs, _ = algorithm
    alg.distillation_iteration = alg.warmup_iterations
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["warmup_student_fraction"] == 0.25
    assert metrics["teacher_walking_candidates"] == 4
    assert metrics["teacher_omni_candidates"] == 4
    assert metrics["teacher_transform_candidates"] == 8
    assert alg.is_warmup


def _rollout_unique_phases(alg, obs, phase):
    with torch.inference_mode():
        for step in range(2):
            current = obs.clone()
            current["actor"][:, 0] = phase + step * 10 + torch.arange(8)
            alg.act(current)
            alg.process_env_step(current, torch.ones(8), torch.zeros(8), {})
        alg.compute_returns(current)


def test_replay_reuses_old_phase_without_ever_optimizing_held_out_samples(algorithm, monkeypatch):
    alg, obs, _ = algorithm
    alg.replay_iterations = 2
    _rollout_unique_phases(alg, obs, 0)
    alg.update()
    assert alg._samples
    old_train = {float(value) for sample in alg._samples for value in sample[0]["actor"][~sample[4], 0]}
    old_validation = {float(value) for sample in alg._samples for value in sample[0]["actor"][sample[4], 0]}
    recorded = []
    original = alg.actor.forward

    def track(batch, *args, **kwargs):
        if torch.is_grad_enabled():
            recorded.extend(batch["actor"][:, 0].tolist())
        return original(batch, *args, **kwargs)

    monkeypatch.setattr(alg.actor, "forward", track)
    monkeypatch.setattr(alg, "_balanced_indices", lambda ids, limit: torch.arange(len(ids)))
    _rollout_unique_phases(alg, obs, 100)
    metrics = alg.update()
    assert old_train <= set(recorded)
    assert old_validation.isdisjoint(recorded)
    assert metrics["distillation_samples"] == 12
    assert metrics["distillation_replay_samples"] == 24
    assert metrics["teacher_walking_candidates"] == 4
    assert metrics["teacher_walking_cached"] == 8


def test_replay_has_bounded_rollout_history_and_is_not_saved(algorithm):
    alg, obs, _ = algorithm
    alg.replay_iterations = 2
    for phase in (0, 100, 200, 300):
        _rollout_unique_phases(alg, obs, phase)
        metrics = alg.update()
        assert metrics["distillation_replay_samples"] <= 24
    assert all(float(sample[0]["actor"][:, 0].min()) >= 200 for sample in alg._samples)
    saved = deepcopy(alg.save())
    assert saved["multi_teacher"]["schedule"]["replay_iterations"] == 2
    assert "samples" not in saved["multi_teacher"]
    alg.load(saved)
    assert not alg._samples
    _rollout_unique_phases(alg, obs, 400)
    alg.update()
    assert alg._samples
    alg.load(saved, {"actor": True})
    assert not alg._samples


def test_replay_allows_teacherless_current_rollout_until_history_expires(algorithm):
    alg, obs, _ = algorithm
    alg.replay_iterations = 2
    rollout(alg, obs)
    alg.update()
    alg.teachers = SimpleNamespace(query=lambda: SimpleNamespace(
        actions=torch.zeros(8, 6), weights=torch.zeros(8), counts={},
    ))
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["distillation_samples"] == 0
    assert metrics["distillation_replay_samples"] == 12
    assert metrics["distillation_coverage"] == 0
    rollout(alg, obs)
    with pytest.raises(RuntimeError, match="没有有效教师"):
        alg.update()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_replay_window_rejects_invalid_configuration(value):
    with pytest.raises(ValueError, match="replay_iterations"):
        MultiTeacherPPO(replay_iterations=value)


def test_joint_training_keeps_independent_ability_metrics(algorithm):
    alg, obs, _ = algorithm
    alg.ppo_start_iteration = 0
    alg.validation_interval_iterations = 1
    alg.validation_callback = lambda: {'validation_passed': False, 'ascent_success': .25}
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics['distillation_warmup'] == 0
    assert metrics['validation_passed'] == 0
    assert metrics['ascent_success'] == .25
    assert 'value' in metrics


def test_anchor_keeps_successful_phase_after_rolling_history_expires(algorithm, monkeypatch):
    alg, obs, _ = algorithm
    alg.replay_iterations = 1
    _rollout_unique_phases(alg, obs, 0)
    alg.update()
    alg._capture_anchor()
    assert alg._anchor is not None
    anchor = alg._anchor
    old_train = set(anchor[0]["actor"][~anchor[4], 0].tolist())
    old_validation = set(anchor[0]["actor"][anchor[4], 0].tolist())
    alg.ppo_start_iteration = alg.distillation_iteration
    recorded = []
    original = alg.actor.forward

    def track(batch, *args, **kwargs):
        if torch.is_grad_enabled():
            recorded.extend(batch["actor"][:, 0].tolist())
        return original(batch, *args, **kwargs)

    monkeypatch.setattr(alg.actor, "forward", track)
    monkeypatch.setattr(alg, "_balanced_indices", lambda ids, limit: torch.arange(len(ids))[:limit])
    # 仅追踪模仿阶段，PPO 自身仍只处理当前真实 rollout。
    from rsl_rl.algorithms import PPO
    monkeypatch.setattr(PPO, "update", lambda self: self.storage.clear() or {})
    _rollout_unique_phases(alg, obs, 100)
    metrics = alg.update()
    assert old_train & set(recorded)
    assert old_validation.isdisjoint(recorded)
    assert all(float(sample[0]["actor"][:, 0].min()) >= 100 for sample in alg._samples)
    assert metrics["distillation_anchor_optimized"] > 0
    assert metrics["distillation_anchor_optimized"] == metrics["distillation_rolling_optimized"]


def test_anchor_is_balanced_cpu_checkpointed_and_cleared_by_hot_start(algorithm):
    alg, obs, _ = algorithm
    alg.anchor_max_samples = 6
    rollout(alg, obs)
    alg.update()
    alg._capture_anchor()
    alg.ppo_start_iteration = 1
    anchor = alg._anchor
    assert anchor is not None
    assert len(anchor[1]) <= 6
    assert torch.bincount(anchor[3], minlength=3).tolist() == [2, 2, 2]
    assert torch.bincount(anchor[3][~anchor[4]], minlength=3).tolist() == [1, 1, 1]
    saved = deepcopy(alg.save())
    assert saved["multi_teacher"]["anchor"][0].device.type == "cpu"
    alg._anchor = None
    alg.load(saved)
    assert not alg.is_warmup
    assert alg._anchor is not None
    torch.testing.assert_close(alg._anchor[1], anchor[1])
    assert not alg._samples
    alg.load(saved, {"actor": True})
    assert alg._anchor is None
    assert alg.is_warmup


def test_legacy_joint_checkpoint_without_anchor_reenters_warmup(algorithm):
    alg, _, _ = algorithm
    alg.ppo_start_iteration = 100
    alg.validation_passes = 3
    saved = deepcopy(alg.save())
    saved["multi_teacher"].pop("anchor", None)
    alg.load(saved)
    assert alg.is_warmup
    assert alg.validation_passes == 0


@pytest.mark.parametrize("value", [0, 2, True, 3.5])
def test_anchor_capacity_rejects_invalid_configuration(value):
    with pytest.raises(ValueError, match="anchor_max_samples"):
        MultiTeacherPPO(anchor_max_samples=value)


def test_enabled_distillation_rejects_zero_weight_even_with_zero_minimum_warmup():
    with pytest.raises(ValueError, match="start_coef 必须大于零"):
        MultiTeacherPPO(warmup_iterations=0, start_coef=0, floor_coef=0)


def test_disabled_distillation_allows_zero_imitation_weight(monkeypatch):
    from rsl_rl.algorithms import PPO

    def reached_base(*args, **kwargs):
        raise RuntimeError("进入纯 PPO 构造")

    monkeypatch.setattr(PPO, "__init__", reached_base)
    with pytest.raises(RuntimeError, match="进入纯 PPO"):
        MultiTeacherPPO(distillation_enabled=False, warmup_iterations=0, start_coef=0, floor_coef=0)


@pytest.mark.parametrize('batch_size', range(1, 6))
def test_imitation_batch_must_cover_three_teachers_and_two_sources(batch_size):
    with pytest.raises(ValueError, match='distillation_batch_size.*6'):
        MultiTeacherPPO(distillation_batch_size=batch_size)


def test_anchor_captures_accepted_student_mean_instead_of_teacher_labels(algorithm):
    alg, obs, _ = algorithm
    rollout(alg, obs)
    alg.update()
    alg._capture_anchor()
    assert alg._anchor is not None
    anchor = alg._anchor
    expected = alg.actor(anchor[0], stochastic_output=False).detach()
    torch.testing.assert_close(anchor[1], expected)
    assert not torch.allclose(anchor[1], torch.full_like(anchor[1], 0.7))


@pytest.mark.parametrize("teacher_coef", [0.0, 1e-9])
def test_anchor_optimization_survives_teacher_weight_annealing(algorithm, monkeypatch, teacher_coef):
    alg, obs, _ = algorithm
    rollout(alg, obs)
    alg.update()
    alg._capture_anchor()
    assert alg._anchor is not None
    anchor = alg._anchor
    alg.ppo_start_iteration = alg.distillation_iteration
    alg.start_coef = alg.floor_coef = teacher_coef
    alg.warmup_target_std = 1
    alg.distillation_updates = 20
    alg.optimizer = torch.optim.Adam(alg.actor.parameters(), lr=0.005)
    with torch.no_grad():
        alg.actor.mlp[-2].bias[:6].add_(1)
    before = (alg.actor(anchor[0], stochastic_output=False) - anchor[1]).square().mean().item()
    from rsl_rl.algorithms import PPO
    monkeypatch.setattr(PPO, "update", lambda self: self.storage.clear() or {})
    rollout(alg, obs)
    metrics = alg.update()
    after = (alg.actor(anchor[0], stochastic_output=False) - anchor[1]).square().mean().item()
    assert after < before
    assert metrics["distillation_anchor_optimized"] > 0
    assert metrics["teacher_omni_anchor_validation_mse"] >= 0
    assert metrics["teacher_omni_anchor_validation_std"] > 0


def test_anchor_protection_coefficient_is_restored_only_for_full_resume(algorithm):
    alg, _, _ = algorithm
    alg.anchor_coef = 17
    saved = deepcopy(alg.save())
    alg.anchor_coef = 23
    alg.load(saved)
    assert alg.anchor_coef == 17
    alg.anchor_coef = 29
    alg.load(saved, {"actor": True})
    assert alg.anchor_coef == 29


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_anchor_coefficient_rejects_invalid_configuration(value):
    with pytest.raises(ValueError, match="anchor_coef"):
        MultiTeacherPPO(anchor_coef=value)


def test_disabled_distillation_skips_even_loaded_anchor(algorithm, monkeypatch):
    alg, obs, _ = algorithm
    rollout(alg, obs)
    alg.update()
    alg._capture_anchor()
    alg.ppo_start_iteration = alg.distillation_iteration
    saved = deepcopy(alg.save())
    alg.distillation_enabled = False
    alg.load(saved)
    assert alg._anchor is not None

    def forbidden(*args, **kwargs):
        raise AssertionError("纯 PPO 不应执行固定或滚动模仿")

    monkeypatch.setattr(alg, "_imitate", forbidden)
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["distillation"] == 0
    assert metrics["distillation_anchor_coefficient"] == 0
    assert "value" in metrics


def test_old_teacher_anchor_cannot_be_used_as_accepted_student_reference(algorithm):
    alg, obs, _ = algorithm
    rollout(alg, obs)
    alg.update()
    alg._capture_anchor()
    alg.ppo_start_iteration = alg.distillation_iteration
    saved = deepcopy(alg.save())
    saved["multi_teacher"].pop("anchor_kind")
    alg.load(saved)
    assert alg._anchor is None
    assert alg.is_warmup
    assert alg.validation_passes == 0


def test_anchor_prefers_successful_validation_trajectories_and_keeps_student_labels(algorithm):
    alg, obs, _ = algorithm
    rollout(alg, obs)
    alg.update()
    validated = obs[:6].clone()
    validated["actor"][:, 0] = torch.arange(40, 46)
    ids = torch.tensor([0, 0, 1, 1, 2, 2])
    held_out = torch.tensor([False, True] * 3)
    alg.anchor_samples_callback = lambda: (validated, torch.full((6, 6), -0.8), torch.ones(6), ids, held_out)
    alg._capture_anchor()
    anchor = alg._anchor
    assert anchor is not None
    assert set(anchor[0]["actor"][:, 0].tolist()) == set(range(40, 46))
    torch.testing.assert_close(anchor[1], alg.actor(anchor[0], stochastic_output=False))
    assert int(anchor[4].sum()) == 3
    saved = deepcopy(alg.save())
    assert saved["multi_teacher"]["anchor_kind"] == "accepted_student_mean"
    assert saved["multi_teacher"]["anchor_observation_source"] == "validation_trajectory"
    alg.ppo_start_iteration = 1
    saved = deepcopy(alg.save())
    alg.load(saved)
    assert alg._anchor_observation_source == "validation_trajectory"
    alg.load(saved, {"actor": True})
    assert alg._anchor_observation_source is None


def test_anchor_callback_none_falls_back_to_rolling_history(algorithm):
    alg, obs, _ = algorithm
    rollout(alg, obs)
    alg.update()
    alg.anchor_samples_callback = lambda: None
    alg._capture_anchor()
    assert alg._anchor is not None
    assert alg.save()["multi_teacher"]["anchor_observation_source"] == "rolling_replay"


@pytest.mark.parametrize('use_mse, expected_warmup', [(True, True), (False, False)])
def test_sensor_student_gate_can_log_unreachable_teacher_mse_without_blocking(algorithm, monkeypatch, use_mse, expected_warmup):
    alg, obs, _ = algorithm
    alg.validation_use_mse = use_mse
    alg.validation_interval_iterations = 1
    alg.validation_required_passes = 1
    alg.validation_min_samples = 1
    alg.validation_max_mse = .02
    alg.validation_max_std = .15
    alg.warmup_iterations = 0
    alg.validation_callback = lambda: {'validation_passed': True}
    def imitate(coefficient):
        metrics = {}
        for name in ('walking', 'omni', 'transform'):
            metrics[f'teacher_{name}_validation_samples'] = 32.
            metrics[f'teacher_{name}_validation_mse'] = .5
            metrics[f'teacher_{name}_validation_std'] = .1
        return .5, metrics
    monkeypatch.setattr(alg, '_imitate', imitate)
    rollout(alg, obs)
    metrics = alg.update()
    assert alg.is_warmup is expected_warmup
    assert metrics['teacher_walking_validation_mse'] == .5
    saved = deepcopy(alg.save())
    alg.validation_use_mse = not use_mse
    alg.load(saved)
    assert alg.validation_use_mse is use_mse


def test_progress_callback_reports_actual_ppo_elapsed_and_resumes(algorithm):
    alg, obs, _ = algorithm
    reported = []
    alg.progress_callback = lambda iteration, pretraining: reported.append((iteration, pretraining))
    rollout(alg, obs)
    alg.update()
    assert reported[-1] == (0, True)
    alg._capture_anchor()
    alg.ppo_start_iteration = alg.distillation_iteration
    rollout(alg, obs)
    alg.update()
    assert reported[-1] == (1, False)
    saved = deepcopy(alg.save())
    alg.distillation_iteration += 50
    alg.load(saved)
    assert reported[-1] == (1, False)
    alg.load(saved, {'actor': True})
    assert reported[-1] == (0, True)


def test_robust_evaluation_reports_separately_and_never_blocks_primary_gate(algorithm):
    alg, obs, _ = algorithm
    alg.robust_validation_interval = 1
    calls = []
    alg.robust_validation_callback = lambda: calls.append(True) or {'validation_passed': False, 'lateral_success': .25}
    rollout(alg, obs)
    alg.update()
    assert not calls
    alg._capture_anchor()
    alg.ppo_start_iteration = alg.distillation_iteration
    rollout(alg, obs)
    metrics = alg.update()
    assert calls == [True]
    assert metrics['robust_validation_passed'] == 0
    assert metrics['robust_lateral_success'] == .25
    assert not alg.is_warmup


@pytest.mark.parametrize('bad_field,bad_value', [('validation_samples', 0), ('validation_std', .3)])
def test_sensor_gate_still_requires_teacher_coverage_and_low_noise(algorithm, monkeypatch, bad_field, bad_value):
    alg, obs, _ = algorithm
    alg.validation_use_mse = False
    alg.validation_interval_iterations = alg.validation_required_passes = alg.validation_min_samples = 1
    alg.warmup_iterations = 0
    alg.validation_callback = lambda: {'validation_passed': True}
    def imitate(coefficient):
        metrics = {}
        for name in ('walking', 'omni', 'transform'):
            metrics[f'teacher_{name}_validation_samples'] = 32.
            metrics[f'teacher_{name}_validation_mse'] = .5
            metrics[f'teacher_{name}_validation_std'] = .1
        metrics[f'teacher_omni_{bad_field}'] = bad_value
        return .5, metrics
    monkeypatch.setattr(alg, '_imitate', imitate)
    rollout(alg, obs)
    metrics = alg.update()
    assert alg.is_warmup
    assert metrics['warmup_blocked_omni'] == 1


def test_runner_publishes_progress_and_wires_primary_and_robust_schema(algorithm, monkeypatch, tmp_path):
    from mjlab.rl import MjlabOnPolicyRunner

    from zbot_rl_mjlab import universal_evaluation
    from zbot_rl_mjlab.distillation import MultiTeacherRunner
    from zbot_rl_mjlab.tasks.universal.env_cfg import universal_env_cfg

    alg, obs, _ = algorithm
    term = SimpleNamespace(tracking_ema=torch.zeros(2), set_pretraining=lambda enabled: None)
    physical = SimpleNamespace(common_step_counter=0, cfg=universal_env_cfg(), command_manager=SimpleNamespace(get_term=lambda name: term))
    env = SimpleNamespace(unwrapped=physical, reset=lambda: (obs, {}))
    def base_init(self, env, cfg, log_dir, device):
        self.alg, self.env, self.cfg = alg, env, cfg
    class Evaluator:
        foundation_samples = None
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def __call__(self, actor):
            return {'validation_passed': False}
    created = []
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', base_init)
    monkeypatch.setattr('zbot_rl_mjlab.teachers.TeacherEnsemble', lambda *args, **kwargs: alg.teachers)
    monkeypatch.setattr(MultiTeacherRunner, '_teacher_provenance', lambda self: {})
    monkeypatch.setattr(universal_evaluation, 'UniversalEvaluator', lambda **kwargs: created.append(Evaluator(**kwargs)) or created[-1])
    runner = MultiTeacherRunner(cast(RslRlVecEnvWrapper, env), {'validation_randomized': False, 'robust_validation_interval': 100, 'console_mode': 'full'}, log_dir=str(tmp_path))
    assert len(created) == 2
    assert created[0].kwargs['env_cfg'] is physical.cfg
    assert not created[0].kwargs['randomized'] and created[1].kwargs['randomized']
    assert physical._student_rl_iteration == 0 and physical._student_pretraining
    alg._capture_anchor = lambda: None
    alg.ppo_start_iteration = 12
    alg.distillation_iteration = 37
    assert alg.progress_callback is not None
    alg.progress_callback(alg.rl_iteration, alg.is_warmup)
    assert physical._student_rl_iteration == 25 and not physical._student_pretraining
    assert runner._run_robust_validation()['validation_passed'] is False
    assert created[1].rl_iteration == 25



def _four_teacher_proxy(alg, parameter, *, include_snake=True):
    names = ("walking", "omni", "transform", "snake")
    ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] if include_snake else [0, 0, 1, 1, 2, 2, 2, 2])
    alg.teachers = cast(TeacherEnsemble, SimpleNamespace(
        teacher_names=names,
        query=lambda: SimpleNamespace(actions=parameter, weights=torch.ones(8),
                                      teacher_ids=ids, counts={name: int((ids == index).sum()) for index, name in enumerate(names)}),
    ))
    return names


def test_four_teacher_sampling_metrics_anchor_and_resume(algorithm):
    alg, obs, parameter = algorithm
    names = _four_teacher_proxy(alg, parameter)
    alg.distillation_max_samples = 16
    rollout(alg, obs)
    assert alg._cached_counts == [4, 4, 4, 4]
    metrics = alg.update()
    for name in names:
        assert metrics[f"teacher_{name}_candidates"] == 4
        assert metrics[f"teacher_{name}_cached"] == 4
        assert metrics[f"teacher_{name}_validation_samples"] == 2
        assert metrics[f"teacher_{name}_optimized"] > 0
    alg.anchor_max_samples = 8
    alg._capture_anchor()
    assert alg._anchor is not None
    assert torch.bincount(alg._anchor[3], minlength=4).tolist() == [2, 2, 2, 2]
    alg.ppo_start_iteration = 1
    saved = deepcopy(alg.save())
    alg.load(saved)
    assert not alg.is_warmup
    assert alg._anchor is not None
    assert (alg._anchor[3] == 3).sum() == 2


def test_four_teacher_gate_and_anchor_require_snake_coverage(algorithm):
    alg, obs, parameter = algorithm
    _four_teacher_proxy(alg, parameter, include_snake=False)
    alg.warmup_iterations = 1
    alg.validation_interval_iterations = 1
    alg.validation_required_passes = 1
    alg.validation_min_samples = 1
    alg.validation_use_mse = False
    alg.validation_max_std = 1
    alg.validation_callback = lambda: {"validation_passed": True}
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["warmup_blocked_snake"] == 1
    assert metrics["warmup_validation_passes"] == 0
    assert alg.is_warmup
    with pytest.raises(RuntimeError, match="缺少教师训练覆盖"):
        alg._capture_anchor()


def test_balanced_sampling_keeps_all_high_anchor_strata():
    labels = torch.arange(8).repeat_interleave(3)
    indices = MultiTeacherPPO._balanced_indices(labels, 16)
    assert len(indices) == 16
    assert labels[indices].bincount(minlength=8).tolist() == [2] * 8


def test_anchor_selection_requires_explicit_fourth_teacher():
    from zbot_rl_mjlab.anchor import select

    obs = TensorDict({"actor": torch.ones(8, 7)}, batch_size=[8])
    held_out = torch.tensor([False, True] * 4)
    ids = torch.arange(4).repeat_interleave(2)
    indices = select(obs, ids, held_out, 8, required_teacher_training=True, expected_teacher_count=4)
    assert set(ids[indices[~held_out[indices]]].tolist()) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="教师 3 缺少训练样本"):
        select(obs[:6], ids[:6], held_out[:6], 8, required_teacher_training=True, expected_teacher_count=4)


def test_four_teacher_gate_accepts_complete_coverage(algorithm):
    alg, obs, parameter = algorithm
    _four_teacher_proxy(alg, parameter)
    alg.distillation_max_samples = 16
    alg.warmup_iterations = 1
    alg.validation_interval_iterations = 1
    alg.validation_required_passes = 1
    alg.validation_min_samples = 1
    alg.validation_use_mse = False
    alg.validation_max_std = 1
    alg.validation_callback = lambda: {"validation_passed": True}
    rollout(alg, obs)
    metrics = alg.update()
    assert metrics["warmup_blocked_snake"] == 0
    assert metrics["warmup_validation_passes"] == 1
    assert not alg.is_warmup
    assert alg._anchor is not None
    assert set(alg._anchor[3].tolist()) == {0, 1, 2, 3}


def test_snake_motion_anchor_has_dedicated_stratum():
    from zbot_rl_mjlab.anchor import strata

    observations = TensorDict({"actor": torch.zeros(2, 185), "critic": torch.zeros(2, 53)}, batch_size=[2])
    observations["critic"][:, 38] = -1
    observations["critic"][:, 41] = 1  # snake phase
    observations["critic"][0, 43] = 0.3
    labels = strata(observations, torch.full((2,), 3))
    assert labels.tolist() == [8, 7]
    indices = MultiTeacherPPO._balanced_indices(torch.arange(9).repeat_interleave(2), 18)
    assert len(indices) == 18
    assert set((indices // 2).tolist()) == set(range(9))
