"""Adaptive tracking weights prioritize weak axes without changing the reward budget."""
from copy import deepcopy
from typing import cast

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from test_site_snake_runner import setup as setup  # noqa: PLC0414

from zbot_rl_mjlab.adaptive_site_snake import (
    VELOCITY_REWARDS,
    AdaptiveSiteSnakeTeacherRunner,
    VelocityWeightBalancer,
)
from zbot_rl_mjlab.experiments.site_snake import site_snake_env_cfg
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunnerCfg
from zbot_rl_mjlab.site_snake_teacher import SiteSnakeTeacherRunner
from zbot_rl_mjlab.tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg

TASK = 'Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher'


def observations(scores=(1., 0., -2.), count=256):
    # Pure-axis exposure, to avoid turning two uncommanded axes into easy tasks.
    raw = torch.zeros(3*count, 61)
    for axis, score in enumerate(scores):
        ids = slice(axis*count, (axis+1)*count)
        raw[ids, 40+axis] = .1
        raw[ids, 49+axis] = .1*(1.+(1.-score)/2.)
    return raw


def test_registered_task_uses_independent_runner_and_checkpoint_role():
    assert load_runner_cls(TASK) is AdaptiveSiteSnakeTeacherRunner
    assert cast(OmniTeacherRunnerCfg, load_rl_cfg(TASK)).class_name.endswith('AdaptiveSiteSnakeTeacherRunner')
    assert AdaptiveSiteSnakeTeacherRunner._metadata_schema()['role'] != SiteSnakeTeacherRunner._metadata_schema()['role']
    fixed = site_snake_env_cfg()
    adaptive = load_env_cfg(TASK)
    assert adaptive.actions == fixed.actions
    assert adaptive.rewards.keys() == fixed.rewards.keys()
    assert {name: adaptive.rewards[name].weight for name in VELOCITY_REWARDS} == {name: fixed.rewards[name].weight for name in VELOCITY_REWARDS}


def test_measure_uses_unweighted_per_axis_scores_and_excludes_stops():
    mean, count = VelocityWeightBalancer.measure(observations(), torch.tensor([.5, .5, .25]))
    torch.testing.assert_close(mean, torch.tensor([1., 0., -2.]))
    torch.testing.assert_close(count, torch.full((3,), 256))


def test_weak_axis_receives_priority_with_constant_budget_and_slow_changes():
    balancer = VelocityWeightBalancer(3.)
    measured = balancer.measure(observations(), torch.tensor([.5, .5, .25]))
    for _ in range(100):
        before = balancer.weights.clone()
        metrics = balancer.advance(measured)
        assert balancer.weights.sum().item() == pytest.approx(3.)
        assert ((balancer.weights >= .5) & (balancer.weights <= 2.)).all()
        assert (balancer.weights-before).abs().max() <= .075+1e-6
        assert metrics['velocity_weight_used_vx'] == before[0].item()
    assert balancer.weights[2] > balancer.weights[1] > balancer.weights[0]


def test_equal_scores_keep_equal_weights_and_unseen_axes_do_not_get_priority():
    balancer = VelocityWeightBalancer(6.)
    for _ in range(5):
        balancer.advance(balancer.measure(observations((.5, .5, .5)), torch.ones(3)))
    torch.testing.assert_close(balancer.weights, torch.full((3,), 2.))
    unobserved = VelocityWeightBalancer(6.)
    raw = observations()[:256]
    for _ in range(5):
        unobserved.advance(unobserved.measure(raw, torch.ones(3)))
    torch.testing.assert_close(unobserved.weights, torch.full((3,), 2.))
    assert unobserved.seen.tolist() == [True, False, False]


def test_tiny_commands_invalid_samples_and_rare_axes_are_ignored():
    raw = observations(count=100)
    raw[:100, 40] = .001
    raw[100:200, 50] = float('nan')
    balancer = VelocityWeightBalancer(3.)
    mean, count = balancer.measure(raw, torch.ones(3))
    assert count.tolist() == [0, 0, 100]
    assert torch.isfinite(mean).all()
    balancer.advance((mean, count))
    assert not balancer.seen.any()
    torch.testing.assert_close(balancer.weights, torch.ones(3))


def test_state_roundtrip_preserves_exact_next_update():
    first = VelocityWeightBalancer(3.)
    measured = first.measure(observations(), torch.ones(3))
    first.advance(measured)
    restored = VelocityWeightBalancer.from_state(first.state_dict())
    assert first.advance(measured) == restored.advance(measured)
    torch.testing.assert_close(first.weights, restored.weights)


@pytest.mark.parametrize('patch', [
    {'version': True}, {'version': 2}, {'budget': float('nan')},
    {'weights': [1., 1., 2.]}, {'weights': [True]*3}, {'weights': [float('nan')]*3},
    {'competency': [0., 1., 2.]}, {'competency': [False]*3},
    {'seen': [1, 0, 1]}, {'updates': -1},
])
def test_invalid_checkpoint_state_is_rejected(patch):
    state = deepcopy(VelocityWeightBalancer(3.).state_dict())
    state.update(patch)
    with pytest.raises(ValueError):
        VelocityWeightBalancer.from_state(state)


@pytest.fixture
def adaptive_runner(setup, tmp_path):
    # Reuse the existing model/runner boundary fixture; no physics or SGD here.
    from types import SimpleNamespace

    env, cfg, alg, *_ = setup
    cfg['validation_interval_iterations'] = 0
    cfg['robust_validation_interval'] = 0
    env.device = 'cpu'
    env.unwrapped.reward_manager = SimpleNamespace(
        terms=deepcopy(env.unwrapped.cfg.rewards))
    env.unwrapped.reward_manager.get_term_cfg = lambda name: env.unwrapped.reward_manager.terms[name]
    raw = observations().unsqueeze(0)
    raw[..., 39] = 2.
    alg.storage = SimpleNamespace(observations={'actor': raw})
    base_update = alg.update
    runner = AdaptiveSiteSnakeTeacherRunner(env, cfg, log_dir=str(tmp_path))
    return runner, env, alg, base_update


def test_runner_changes_live_weights_after_update_even_when_storage_is_cleared(adaptive_runner):
    runner, env, alg, base_update = adaptive_runner
    before = runner.velocity_balancer.weights.clone()
    manager = env.unwrapped.reward_manager

    def update():
        torch.testing.assert_close(runner.velocity_balancer.weights, before)
        assert [manager.get_term_cfg(n).weight for n in VELOCITY_REWARDS] == before.tolist()
        alg.storage.observations['actor'].zero_()
        return {'surrogate': .2}

    base_update.side_effect = update
    metrics = alg.update()
    assert metrics['surrogate'] == .2
    assert runner.velocity_balancer.weights[2] > runner.velocity_balancer.weights[0]
    assert [manager.get_term_cfg(n).weight for n in VELOCITY_REWARDS] == runner.velocity_balancer.weights.tolist()
    assert [env.unwrapped.cfg.rewards[n].weight for n in VELOCITY_REWARDS] == runner.velocity_balancer.weights.tolist()


def test_runner_defers_snapshot_until_adaptive_state_is_committed(adaptive_runner, monkeypatch, tmp_path):
    runner, _, alg, base_update = adaptive_runner
    captured = []
    monkeypatch.setattr(SiteSnakeTeacherRunner, 'save',
                        lambda self, path, infos=None: captured.append(infos))
    def update():
        runner.save(tmp_path/'qualified.pt', {'qualification': True})
        assert not captured
        return {'surrogate': .2}

    base_update.side_effect = update
    alg.update()
    assert captured[0]['adaptive_velocity_weights']['updates'] == 1
    assert captured[0]['qualification']


def test_invalid_balancer_state_is_rejected_before_parent_model_load(adaptive_runner, monkeypatch, tmp_path):
    from unittest.mock import Mock

    runner, _, _, _ = adaptive_runner
    state = runner.velocity_balancer.state_dict()
    state['weights'] = [float('nan')]*3
    saved = {'infos': {'privileged_omni_teacher': runner._schema(), 'adaptive_velocity_weights': state}}
    path = tmp_path/'bad.pt'
    torch.save(saved, path)
    load = Mock()
    monkeypatch.setattr(SiteSnakeTeacherRunner, 'load', load)
    with pytest.raises(ValueError, match='adaptive velocity state'):
        runner.load(path)
    load.assert_not_called()


@pytest.mark.parametrize('load_cfg,resumes_balance', [(None, True), ({'actor': True}, False)])
def test_load_restores_adaptation_only_for_full_training_resume(adaptive_runner, monkeypatch, tmp_path, load_cfg, resumes_balance):
    runner, env, _, _ = adaptive_runner
    other = VelocityWeightBalancer(runner.velocity_balancer.budget)
    other.advance(other.measure(observations(), torch.ones(3)))
    saved = {'infos': {'privileged_omni_teacher': runner._schema(),
                       'adaptive_velocity_weights': other.state_dict()}}
    path = tmp_path/'valid.pt'
    torch.save(saved, path)
    monkeypatch.setattr(SiteSnakeTeacherRunner, 'load', lambda *args: saved['infos'])
    initial = runner.velocity_balancer.state_dict()
    runner.load(path, load_cfg)
    assert runner.velocity_balancer.state_dict() == (other.state_dict() if resumes_balance else initial)
    if resumes_balance:
        assert [env.unwrapped.reward_manager.get_term_cfg(n).weight for n in VELOCITY_REWARDS] == other.weights.tolist()


def test_playback_restores_saved_control_and_weights_without_updates(adaptive_runner, tmp_path):
    runner, _, _, _ = adaptive_runner
    runner.velocity_balancer.advance(runner.velocity_balancer.measure(observations(), torch.ones(3)))
    saved = {'infos': {'privileged_omni_teacher': runner._schema(),
                       'adaptive_velocity_weights': runner.velocity_balancer.state_dict()}}
    path = tmp_path/'play.pt'
    torch.save(saved, path)
    cfg = load_env_cfg(TASK, play=True)
    AdaptiveSiteSnakeTeacherRunner.prepare_playback_env_cfg(cfg, path)
    assert [cfg.rewards[n].weight for n in VELOCITY_REWARDS] == runner.velocity_balancer.weights.tolist()
    assert cast(SnakeCrawlJointTargetRateActionCfg, cfg.actions['joint_pos']).target_limit == saved['infos']['privileged_omni_teacher']['joint_target_limit']
    with pytest.raises(ValueError, match='compatible'):
        SiteSnakeTeacherRunner._validate_checkpoint(saved)


def test_all_stop_rollout_freezes_already_initialized_weights():
    balancer = VelocityWeightBalancer(3.)
    balancer.advance(balancer.measure(observations(), torch.ones(3)))
    weights, competency = balancer.weights.clone(), balancer.competency.clone()
    balancer.advance(balancer.measure(torch.zeros(512, 61), torch.ones(3)))
    torch.testing.assert_close(balancer.weights, weights)
    torch.testing.assert_close(balancer.competency, competency)
