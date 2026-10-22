"""Balance realized reward contributions rather than imposing a weight ratio."""
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from zbot_rl_mjlab.adaptive_site_snake import VELOCITY_REWARDS, VelocityWeightBalancer
from zbot_rl_mjlab.base_snake_teacher import (
    BaseSnakeAdaptiveTeacherRunner,
    RewardContributionBalancer,
)


def step(balancer, raw, counts=None):
    raw = torch.tensor(raw)
    balancer.record_reward_means(raw)
    return balancer.advance((raw, torch.full((3,), 256) if counts is None else counts))


def contribution_ratio(balancer, raw):
    contribution = balancer.weights*torch.tensor(raw)
    return (contribution[2]/contribution[:2].sum()).item()


def assert_budget(balancer):
    assert balancer.weights.sum().item() == pytest.approx(balancer.budget)
    assert balancer._weights_within_bounds(balancer.weights, balancer.budget*1e-6)
    assert torch.isfinite(balancer.weights).all() and (balancer.weights > 0).all()


def test_initial_weights_prioritize_planar_learning():
    balancer = RewardContributionBalancer(9.)
    torch.testing.assert_close(balancer.weights, torch.tensor([4.275, 4.275, .45]))
    assert balancer.stage == 1
    assert balancer.weights[2].item()/balancer.budget == pytest.approx(.05)
    torch.testing.assert_close(VelocityWeightBalancer(3.).weights, torch.ones(3))


@pytest.mark.parametrize('raw', [[.1, .05, 1.], [.3, .1, .9], [1., 1., 1.]])
def test_both_positive_reward_targets_converge(raw):
    balancer = RewardContributionBalancer(9.)
    for _ in range(400):
        step(balancer, raw)
        assert_budget(balancer)
    assert contribution_ratio(balancer, raw) == pytest.approx(.25, rel=1e-3)
    returns = balancer.weights*torch.tensor(raw)
    assert returns[0].item() == pytest.approx(returns[1].item(), rel=1e-3)
    if raw[2] > max(raw[:2]):
        assert balancer.weights[2] < 1.5
        assert balancer.weights[1] > balancer.weights[0]


def test_negative_yaw_reward_keeps_its_weight_without_yaw_command_exposure():
    balancer = RewardContributionBalancer(3.)
    before = balancer.weights[2].item()
    step(balancer, [.1, .1, -1.], torch.tensor([256, 256, 0]))
    assert not balancer.seen[2] and balancer.weights[2].item() == pytest.approx(before)


def test_all_stop_and_tiny_rewards_do_not_manufacture_adjustments():
    balancer = RewardContributionBalancer(3.)
    initial = balancer.weights.clone()
    step(balancer, [1., 1., 1.], torch.zeros(3, dtype=torch.long))
    assert not balancer.reward_mean_initialized
    torch.testing.assert_close(balancer.weights, initial)
    step(balancer, [1e-12]*3)
    torch.testing.assert_close(balancer.weights, initial)


@pytest.mark.parametrize('raw', [[0., 0., 1.], [-1., -.5, 1.]])
def test_positive_yaw_is_suppressed_until_planar_return_is_positive(raw):
    balancer = RewardContributionBalancer(3.)
    for _ in range(400):
        step(balancer, raw)
        assert_budget(balancer)
    assert balancer.stage == 1
    assert balancer.weights[2].item() == pytest.approx(3*.05, rel=1e-3)


@pytest.mark.parametrize('raw', [[1., 1., 0.], [1., 1., -1.]])
def test_second_stage_strengthens_yaw_signal_until_yaw_is_positive(raw):
    balancer = RewardContributionBalancer(3.)
    for _ in range(400):
        step(balancer, raw)
        assert_budget(balancer)
    assert balancer.stage == 2
    assert balancer.weights[2].item()/balancer.budget == pytest.approx(.4, rel=1e-3)


def test_positive_ratio_outside_weight_bounds_is_reported():
    balancer = RewardContributionBalancer(3.)
    for _ in range(400):
        step(balancer, [1., 1., .03])
    assert balancer.stage == 2 and balancer.yaw_target_limited
    assert_budget(balancer)


def test_checkpoint_roundtrip_preserves_next_reward_feedback_update():
    original = RewardContributionBalancer(3.)
    step(original, [.1, .2, 1.])
    restored = RewardContributionBalancer.from_state(original.state_dict())
    assert step(original, [.2, .3, 1.]) == step(restored, [.2, .3, 1.])
    torch.testing.assert_close(original.reward_mean_ema, restored.reward_mean_ema)
    assert_budget(restored)


@pytest.mark.parametrize('patch', [
    {'version': 1}, {'yaw_to_planar_reward_target': .5},
    {'weights': [.1, 1., 1.9]}, {'reward_mean_ema': [float('nan')]*3},
    {'reward_mean_initialized': 1}, {'forward_to_lateral_reward_target': .5},
])
def test_bad_contribution_checkpoint_state_is_rejected(patch):
    state = RewardContributionBalancer(3.).state_dict()
    state.update(patch)
    with pytest.raises(ValueError):
        RewardContributionBalancer.from_state(state)


def test_capture_uses_actual_weighted_buffer_before_reset_and_old_rollout_weights():
    runner: Any = object.__new__(BaseSnakeAdaptiveTeacherRunner)
    runner.velocity_balancer = RewardContributionBalancer(3.)
    weights = runner.velocity_balancer.weights.clone()
    rates = torch.tensor([[-.1, -.2, -.9], [.3, .4, .7]])*weights
    manager = SimpleNamespace(active_terms=list(VELOCITY_REWARDS)+['failure_reset'],
                              _step_reward=torch.zeros(2, 4))

    def compute(dt):
        manager._step_reward[:, :3] = rates
        manager._step_reward[:, 3] = -100.
        return manager._step_reward.sum(-1)*dt

    manager.compute = compute
    cfg = SimpleNamespace(commands={'transform': SimpleNamespace(velocity_limits=(.2, .2, .5))})
    terms = {name: SimpleNamespace(params={'std_fraction': .25}) for name in VELOCITY_REWARDS}
    manager.get_term_cfg = lambda name: terms[name]
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(reward_manager=manager, cfg=cfg))
    runner._velocity_reward_sum = torch.zeros(3)
    runner._velocity_reward_steps = 0
    runner._install_velocity_reward_capture()
    total = manager.compute(.02)
    torch.testing.assert_close(total, (rates.sum(-1)-100.)*.02)
    manager._step_reward.zero_()  # Simulate manager/reset buffer changes after capture.
    raw = torch.zeros(256, 61)
    raw[:, 40:43] = .1
    raw[:, 49:52] = .1  # Pre-action observation scores differ from actual collected scores.
    measured = runner._measure_velocity(raw, torch.tensor([.2, .2, .5]))
    metrics = runner._advance_velocity_weights(measured)
    torch.testing.assert_close(runner.velocity_balancer.reward_mean_ema, torch.tensor([.1, .1, -.1]))
    assert metrics['velocity_reward_ratio_valid'] == 0.
    assert metrics['velocity_yaw_to_planar_reward_ratio'] == 0.
    assert metrics['velocity_reward_ratio_target'] == .25
    assert math.isfinite(metrics['velocity_yaw_to_planar_reward_ratio'])
    assert metrics['velocity_reward_yaw_mean'] == pytest.approx(rates[:, 2].mean().item())

@pytest.mark.parametrize('values', [[-2.1, 0., 1.], [0., 0., 1.1], [0., float('nan'), 0.]])
def test_invalid_signed_feedback_is_rejected(values):
    with pytest.raises(ValueError):
        RewardContributionBalancer(3.).record_reward_means(torch.tensor(values))


def test_planar_sign_uses_net_return_including_negative_axis():
    balancer = RewardContributionBalancer(3.)
    for _ in range(400):
        step(balancer, [.2, -1., 1.])
    # A positive forward score cannot conceal the larger lateral penalty.
    assert balancer.stage == 1
    assert balancer.weights[2].item() == pytest.approx(.15, rel=1e-3)
    torch.testing.assert_close(balancer.reward_mean_ema, torch.tensor([.2, -1., 1.]))


def test_signed_feedback_recovers_from_suppression_when_planar_learns():
    balancer = RewardContributionBalancer(3.)
    for raw in ([-1., -1., 1.], [.4, .4, 1.]):
        for _ in range(400):
            step(balancer, raw)
    assert contribution_ratio(balancer, [.4, .4, 1.]) == pytest.approx(.25, rel=1e-3)
    assert not balancer.yaw_target_limited


@pytest.mark.parametrize('raw', [[1., .001, 1.], [.001, 1., 1.]])
def test_first_stage_favors_weaker_planar_axis_without_promoting_yaw(raw):
    balancer = RewardContributionBalancer(3.)
    for _ in range(400):
        step(balancer, raw)
        assert_budget(balancer)
    assert balancer.stage == 1
    weak = 0 if raw[0] < raw[1] else 1
    assert balancer.weights[weak] > balancer.weights[1-weak]
    assert balancer.weights[2].item() == pytest.approx(.15)


def test_mixed_sign_planar_rewards_prioritize_negative_axis_in_first_stage():
    balancer = RewardContributionBalancer(3.)
    for _ in range(100):
        step(balancer, [-1., .2, 1.])
    assert balancer.stage == 1
    assert balancer.weights[0] > balancer.weights[1]


def test_transition_requires_twenty_consecutive_covered_positive_updates():
    balancer = RewardContributionBalancer(3.)
    for _ in range(19):
        step(balancer, [.3, .3, -.5])
    assert balancer.stage == 1 and balancer.planar_positive_updates == 19
    step(balancer, [.3, .3, -.5], torch.tensor([256, 0, 256]))
    assert balancer.planar_positive_updates == 0
    for _ in range(19):
        step(balancer, [.3, .3, -.5])
    assert balancer.stage == 1
    step(balancer, [.3, .3, -.5])
    assert balancer.stage == 2
    assert balancer.weights[2].item() > .15


def test_positive_planar_sum_cannot_hide_one_negative_axis():
    balancer = RewardContributionBalancer(3.)
    for _ in range(100):
        step(balancer, [1., -.1, 1.])
    assert balancer.stage == 1


def test_sustained_planar_regression_returns_to_first_stage():
    balancer = RewardContributionBalancer(3.)
    for _ in range(100):
        step(balancer, [.3, .3, -.5])
    assert balancer.stage == 2
    for _ in range(200):
        step(balancer, [.3, -.5, -.5])
    assert balancer.stage == 1
    assert balancer.weights[2].item() == pytest.approx(.15, rel=1e-3)
    assert balancer.weights[1] > balancer.weights[0]


def test_checkpoint_restores_partial_stage_transition_counter():
    original = RewardContributionBalancer(3.)
    for _ in range(19):
        step(original, [.3, .3, -.5])
    restored = RewardContributionBalancer.from_state(original.state_dict())
    assert step(original, [.3, .3, -.5]) == step(restored, [.3, .3, -.5])
    assert original.stage == restored.stage == 2


@pytest.mark.parametrize('patch', [
    {'stage': 3}, {'stage': True}, {'planar_positive_updates': 20},
    {'planar_regression_updates': -1}, {'stage': 2, 'planar_positive_updates': 1},
    {'stage': 1, 'planar_regression_updates': 1},
])
def test_invalid_stage_checkpoint_rejected(patch):
    state = RewardContributionBalancer(3.).state_dict() | patch
    with pytest.raises(ValueError, match='curriculum'):
        RewardContributionBalancer.from_state(state)
