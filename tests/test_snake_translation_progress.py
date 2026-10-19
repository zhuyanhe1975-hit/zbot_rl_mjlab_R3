"""平移停滞状态、奖励拆分和六方向统计的回归边界。"""

import math

import pytest
import torch
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


def old_score(actual, requested):
    """独立保存拆分前的平方误差改善公式，避免用新分项充当预期值。"""
    tolerance = actual.new_tensor((.04, .04, .20))
    scale = torch.maximum(requested.abs(), tolerance)
    active = requested.abs() > .001
    improvement = (2 * actual * requested - actual.square()) / scale.square()
    movement = (improvement * active).sum(-1) / active.sum(-1).clamp_min(1)
    drift = torch.where(active, 0., (actual / tolerance).square()).sum(-1)
    return torch.where(active.any(-1), movement - .05 * drift.clamp(max=20), 0.)


def progress_state(n=1):
    return mdp.TranslationProgressState(n, 'cpu', snake_env_cfg(play=True))


def request(n=1, vx=.05, vy=0., yaw=0.):
    return torch.tensor([[vx, vy, yaw]]).repeat(n, 1)


def test_score_components_match_independent_old_formula():
    generator = torch.Generator().manual_seed(107)
    actual = torch.randn(512, 3, generator=generator) * .12
    requested = torch.randn(512, 3, generator=generator) * .1
    requested[::2, 0] = 0
    requested[::3, 1] = 0
    requested[::5, 2] = 0
    requested[:4] = torch.tensor([[0., 0., 0.], [.001, -.001, 0.], [.05, .05, .3], [.05, 0., 0.]])
    translation, yaw, drift = mdp.tracking_components(actual, requested)
    expected = old_score(actual, requested)
    torch.testing.assert_close(5 * translation + 5 * yaw - .25 * drift, 5 * expected)
    torch.testing.assert_close(mdp.tracking_score(actual, requested), expected)
    assert drift[0].item() == 0
    assert drift[1].item() == 0


@pytest.mark.parametrize('actual', [[.05, 0., -.1], [.05, 0., -.3], [0., .02, 0.]])
@pytest.mark.parametrize('grounded', [True, False])
def test_legacy_diagnostic_score_retains_original_ground_gate(monkeypatch, actual, grounded):
    env = state_env()
    requested = request(2, vx=.05, yaw=.3)
    actual = torch.tensor([actual]).repeat(2, 1)
    env.command_manager.get_term('motion').command.copy_(requested)
    monkeypatch.setattr(mdp, 'planar_velocity', lambda env: actual)
    monkeypatch.setattr(mdp, 'com_state', lambda env: torch.tensor([[.05, 0.], [.05, 0.]]))
    monkeypatch.setattr(mdp, 'ground_support', lambda env: torch.full((2,), 1. if grounded else 0.))
    expected = old_score(actual, requested)
    if not grounded:
        expected = expected.clamp_max(0)
    combined = 5 * mdp.velocity_tracking(env)
    torch.testing.assert_close(combined, 5 * expected)


def test_stationary_progress_has_grace_then_full_cost_at_one_second():
    state, cmd = progress_state(), request()
    stationary = torch.zeros_like(cmd)
    state.update(stationary, cmd, cmd, 0, .02)
    for step in range(1, 50):
        assert state.update(stationary, cmd, cmd, step, .02).item() == 0
    assert state.age.item() == pytest.approx(.98)
    assert state.update(stationary, cmd, cmd, 50, .02).item() == 1
    assert state.age.item() == pytest.approx(1.)


def test_request_snapshot_survives_inplace_changes_and_equal_values_preserve_age():
    state, cmd = progress_state(), request()
    state.update(cmd, cmd, cmd, 0, .1)
    for step in range(1, 5):
        state.update(cmd, cmd, cmd.clone(), step, .1)
    assert state.age.item() == pytest.approx(.4)
    cmd[:, 0] = -.05
    state.update(cmd, cmd, cmd, 5, .1)
    assert state.age.item() == 0
    assert state.progress.item() == 0
    assert state.request[0, 0].item() == pytest.approx(-.05)
    state.update(cmd, cmd, cmd, 6, .1)
    assert state.age.item() == pytest.approx(.1)


def test_smoothing_changes_do_not_restart_stable_request_age():
    state, target = progress_state(), request()
    for step in range(14):
        smoothed = target * (.4 + step * .04)
        state.update(torch.zeros_like(target), smoothed, target, step, .1)
    assert state.age.item() == pytest.approx(1.3)
    assert state.cost.item() == 1


def test_repeated_reads_and_partial_reset_do_not_advance_other_environments():
    state, cmd = progress_state(2), request(2)
    state.update(cmd, cmd, cmd, 0, .1)
    state.update(cmd, cmd, cmd, 1, .1)
    before = (state.age.clone(), state.progress.clone(), state.cost.clone())
    state.update(-cmd, cmd, cmd, 1, .1)
    for actual, expected in zip((state.age, state.progress, state.cost), before, strict=True):
        torch.testing.assert_close(actual, expected)
    state.reset(torch.tensor([0]))
    assert state.age[0].item() == 0
    assert state.progress[0].item() == 0
    assert state.cost[0].item() == 0
    assert torch.isnan(state.request[0]).all()
    state.update(cmd, cmd, cmd, 1, .1)
    assert state.age.tolist() == pytest.approx([0., .1])
    assert state.progress[1].item() == pytest.approx(before[1][1].item())
    state.update(cmd, cmd, cmd, 2, .1)
    assert state.age.tolist() == pytest.approx([.1, .2])


@pytest.mark.parametrize('vx,vy,yaw', [(0., 0., 0.), (0., 0., .3), (.00999, 0., 0.)])
def test_parking_yaw_and_subthreshold_requests_never_incur_stall(vx, vy, yaw):
    state, cmd = progress_state(), request(vx=vx, vy=vy, yaw=yaw)
    for step in range(30):
        assert state.update(torch.zeros_like(cmd), cmd, cmd, step, .1).item() == 0


@pytest.mark.parametrize('requested_speed,command_speed,expected', [(.01, .01, 1.), (.01, .00999, 0.), (.00999, .01, 0.)])
def test_minimum_speed_threshold_is_inclusive(requested_speed, command_speed, expected):
    state = progress_state()
    requested, cmd = request(vx=requested_speed), request(vx=command_speed)
    for step in range(12):
        state.update(torch.zeros_like(cmd), cmd, requested, step, .1)
    assert state.cost.item() == expected


@pytest.mark.parametrize('ratio,expected', [(.2, 0.), (.1, .5), (0., 1.), (-.2, 1.)])
def test_exact_progress_boundary_and_hinge(ratio, expected):
    state, cmd = progress_state(), request()
    state.update(cmd, cmd, cmd, 0, .1)
    state.age[:] = 1.
    state.progress[:] = ratio
    cost = state.update(cmd * ratio, cmd, cmd, 1, .1)
    assert cost.item() == pytest.approx(expected, abs=1e-6)


def test_large_motion_pulse_is_clipped_before_filtering():
    state, cmd = progress_state(), request()
    state.update(cmd, cmd, cmd, 0, .1)
    state.update(cmd * 10000, cmd, cmd, 1, .1)
    assert state.progress.item() == pytest.approx(1 - math.exp(-.1 / .5))
    for step in range(2, 12):
        state.update(torch.zeros_like(cmd), cmd, cmd, step, .1)
    assert state.cost.item() > .8


@pytest.mark.parametrize('vx,vy', [(-.05, 0.), (0., .05), (0., -.05)])
def test_backward_or_sideways_motion_does_not_satisfy_forward_progress(vx, vy):
    state, cmd = progress_state(), request()
    actual = request(vx=vx, vy=vy)
    for step in range(12):
        state.update(actual, cmd, cmd, step, .1)
    assert state.cost.item() == 1
    assert state.progress.item() <= 0


def test_progress_uses_requested_direction_and_scale_during_command_smoothing():
    state = progress_state()
    requested = request(vx=0., vy=.05)
    smoothed = request(vx=.04, vy=.01)
    forward_inertia = request(vx=.04)
    for step in range(12):
        state.update(forward_inertia, smoothed, requested, step, .1)
    assert state.progress.item() == 0
    assert state.cost.item() == 1
    state.progress[:] = .2
    side_progress = request(vx=0., vy=.01)
    state.update(side_progress, smoothed, requested, 12, .1)
    assert state.progress.item() == pytest.approx(.2)
    assert state.cost.item() == pytest.approx(0., abs=1e-6)


def test_reward_and_metrics_share_one_state_update():
    env = state_env()
    env.common_step_counter = 0
    term = env.command_manager.get_term('motion')
    term.command[:, 0] = .05
    term.requested_velocity.copy_(term.command)
    mdp.translation_stall_cost(env)
    env.common_step_counter = 1
    mdp.translation_stall_cost(env)
    before = mdp.translation_progress_state(env).age.clone()
    mdp.translation_progress_ratio(env)
    mdp.translation_stall_fraction(env)
    mdp.translation_stall_cost(env)
    torch.testing.assert_close(mdp.translation_progress_state(env).age, before)


@pytest.mark.parametrize('axis,sign', [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)])
def test_direction_metrics_use_selected_group_and_signed_projection(monkeypatch, axis, sign):
    env = state_env()
    term = env.command_manager.get_term('motion')
    term.requested_velocity[0, axis] = sign * .05
    term.command.copy_(term.requested_velocity)
    actual = torch.full((2, 3), 9.)
    actual[0, axis] = sign * .03
    monkeypatch.setattr(mdp, 'planar_velocity', lambda env: actual)
    assert mdp.direction_motion_metric(env, axis, sign).tolist() == pytest.approx([.03, .03])
    assert mdp.direction_motion_metric(env, axis, sign, error=True).tolist() == pytest.approx([.02, .02])
    assert mdp.direction_motion_metric(env, axis, -sign).eq(0).all()
    term.requested_velocity[0, (axis + 1) % 3] = .02
    assert mdp.direction_motion_metric(env, axis, sign).eq(0).all()


def test_stall_is_diagnostic_only_and_preserves_observations_and_physical_safety():
    env = state_env()
    cfg = env.cfg
    assert set(cfg.terminations) == {'unsafe', 'head_tilt', 'self_collision', 'time_out'}
    assert set(cfg.events) == {'reset_scene', 'reset_snake'}
    assert mdp.actor_state(env).shape == (2, 40)
    assert mdp.critic_state(env).shape == (2, 55)
    assert 'translation_stall' not in cfg.rewards
    assert 'translation_stall_fraction' not in cfg.metrics
    assert set(cfg.rewards) == {'forward_speed', 'lateral_speed_cost', 'vertical_speed_cost', 'joint_pose_deviation', 'head_tilt_reset_cost', 'target_saturation_fraction'}
    assert cfg.rewards['forward_speed'].weight == 1.
    assert cfg.rewards['joint_pose_deviation'].weight == -.01
    assert 'velocity_tracking' not in cfg.rewards


@pytest.mark.parametrize('field,value', [('stall_min_speed', 0.), ('stall_filter_s', float('nan')), ('stall_grace_s', -1.), ('stall_progress_fraction', 0.), ('stall_progress_fraction', 1.01)])
def test_invalid_progress_parameters_fail_before_state_creation(field, value):
    cfg = snake_env_cfg(play=True)
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        mdp.TranslationProgressState(1, 'cpu', cfg)
