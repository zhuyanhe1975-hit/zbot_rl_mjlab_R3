"""The new teacher follows site commands without inheriting world-X constraints."""
import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.tasks.snake import site_omni as task
from zbot_rl_mjlab.tasks.universal import omni
from zbot_rl_mjlab.tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg
from zbot_rl_mjlab.tasks.universal.privileged import (
    PrivilegedOmniCommandCfg,
    privileged_state,
)


def reward_env(monkeypatch, actual, targets):
    actual, targets = torch.tensor(actual), torch.tensor(targets)
    term = SimpleNamespace(velocity_command=targets)
    monkeypatch.setattr(task.mdp, 'command_term', lambda env: term)
    monkeypatch.setattr(task.mdp, 'tracking_velocity', lambda env: actual)
    position = torch.zeros(len(actual), 6)
    return SimpleNamespace(
        cfg=SimpleNamespace(actions={'joint_pos': SimpleNamespace(snake_target_limit=math.pi/3)},
                            sim=SimpleNamespace(mujoco=SimpleNamespace(gravity=(0., 0., -9.81))),
                            site_up_axis=(1., 0., 0.)),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(
            joint_pos=position, site_quat_w=torch.tensor([2**-.5, 0., -2**-.5, 0.]).repeat(len(actual), 1, 1)))},
        action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(commanded_target=position)),
        _omni_motion_site_index=0,
    )


@pytest.mark.parametrize('direction', [(1., 0.), (-1., 0.), (0., 1.), (0., -1.), (.6, .8)])
@pytest.mark.parametrize('speed', [.03, .05, .08, .1, .15, .2])
def test_linear_kernel_peaks_at_requested_speed_and_rejects_reverse_and_overspeed(monkeypatch, direction, speed):
    x, y = direction
    env = reward_env(monkeypatch, [[s*x, s*y, 0.] for s in (0., speed, -speed, 2*speed)],
                     [[speed*x, speed*y, 0.]]*4)
    reward = task.linear_tracking(env)
    assert reward[1].item() == pytest.approx(1.)
    assert reward[1] > reward[0] > reward[2]
    torch.testing.assert_close(reward[0], reward[3])
    # Even a low-speed command must distinguish stopping from tracking.
    assert reward[0] < .5


def test_linear_kernel_penalizes_perpendicular_drift_including_pure_turn_and_stop(monkeypatch):
    target = [[.1, 0., 0.], [0., 0., .3], [0., 0., 0.]]
    env = reward_env(monkeypatch, [[.1, 0., 0.], [0., 0., .3], [0., 0., 0.]], target)
    torch.testing.assert_close(task.linear_tracking(env), torch.ones(3))
    env = reward_env(monkeypatch, [[.1, .05, 0.], [.05, 0., .3], [.05, 0., 0.]], target)
    assert (task.linear_tracking(env) < 1).all()


@pytest.mark.parametrize('sign', [-1., 1.])
def test_angular_kernel_peaks_at_signed_request_without_heading_lock(monkeypatch, sign):
    env = reward_env(monkeypatch, [[0., 0., w*sign] for w in (.4, 0., -.4)], [[0., 0., .4*sign]]*3)
    reward = task.angular_tracking(env)
    assert reward[0].item() == pytest.approx(1.)
    assert reward[0] > reward[1] > reward[2]


def test_archived_yaw_width_is_used_for_both_translation_and_pure_turn(monkeypatch):
    env = reward_env(monkeypatch, [[.1, 0., .25], [0., 0., .25], [.1, 0., 0.]],
                     [[.1, 0., 0.], [0., 0., 0.], [.1, 0., 0.]])
    torch.testing.assert_close(task.angular_tracking(env), torch.tensor([math.exp(-1), math.exp(-1), 1.]))


@pytest.mark.parametrize('play', [False, True])
def test_config_is_independent_snake_with_archived_physics_and_universal_privileged_layout(play):
    cfg = task.site_snake_env_cfg(play)
    command = cast(PrivilegedOmniCommandCfg, cfg.commands['transform'])
    action = cast(SnakeCrawlJointTargetRateActionCfg, cfg.actions['joint_pos'])
    assert cfg.site_snake_omni and cfg.omni_site_frame and cfg.motion_site == 'foot_0_sole'
    assert command.mode == 'snake' and command.task_observation_encoding == 'scalar'
    assert command.play_random_velocity == play and command.training != play
    assert not command.train_auto_recovery and not command.play_auto_recovery
    assert command.resampling_time_range == (5., 9.)
    assert command.velocity_limits == (.2, .2, .5)
    assert command.motion_axis_min_fraction == .4
    assert action.target_limit == action.snake_target_limit == pytest.approx(math.pi/3)
    assert action.max_target_speed == pytest.approx(2*math.pi)
    assert action.biped_residual_scale is action.transition_end_joint_radius is None
    assert cfg.observations['actor'].terms['privileged'].func is privileged_state
    assert set(cfg.events) == {'reset_scene', 'reset_transition'}
    assert cfg.events['reset_transition'].func is omni.reset_omni
    assert not any('energy' in name or 'yaw_integral' in name or 'transition' in name for name in cfg.rewards)
    assert cfg.sim.mujoco.timestep == .005 and cfg.decimation == 4
    assert cfg.episode_length_s == 30. and cfg.auto_reset != play
    assert {s.name for s in cfg.scene.sensors} == {'base_ground_contact', 'feet_ground_contact', 'transform_ground_contact', 'snake_self_contact'}
    assert set(cfg.terminations) == (set() if play else {'self_collision', 'site_tilt'})
    if not play:
        assert cfg.terminations['self_collision'].func is task.snake_mdp.self_collision
    assert set(cfg.rewards) == {'track_linear_velocity', 'track_angular_velocity',
                                'joint_pos_limits', 'joint_target_limits', 'site_tilt'}
    assert cfg.rewards['joint_pos_limits'].weight == -1
    assert cfg.rewards['joint_target_limits'].weight == -1
    assert cfg.rewards['track_linear_velocity'].weight == 4
    assert cfg.rewards['track_angular_velocity'].weight == 1
    assert 'site_height' in cfg.metrics and 'com_height' not in cfg.metrics


@pytest.mark.parametrize('play', [False, True])
def test_every_reset_and_timer_keeps_all_robots_snake_and_samples_every_axis(monkeypatch, play):
    class Scene(dict):
        env_origins: torch.Tensor
    torch.manual_seed(71)
    cfg = task.site_snake_env_cfg(play)
    n = 512
    roots = torch.zeros(n, 13)
    roots[:, 3] = 1
    q = torch.full((n, 6), 99.)
    robot = SimpleNamespace(data=SimpleNamespace(default_root_state=roots.clone()),
                            write_joint_state_to_sim=lambda pos, vel, env_ids: q.__setitem__(env_ids, pos),
                            write_root_state_to_sim=lambda root, env_ids: roots.__setitem__(env_ids, root))
    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(n, 3)
    env = SimpleNamespace(cfg=cfg, num_envs=n, device='cpu', step_dt=.02, common_step_counter=0, scene=scene)
    term = cast(PrivilegedOmniCommandCfg, cfg.commands['transform']).build(env)
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    monkeypatch.setattr(omni, 'collision_radial_envelope', lambda pose: ([], -.26, .26))
    monkeypatch.setattr(task.mdp, 'configuration_ready', lambda env, relaxed=False: torch.ones(n, dtype=torch.bool))
    omni.reset_omni(env, None)
    term.reset(torch.arange(n))
    assert not q.any() and not term.standing_mode.any()
    assert (term.current_task == 2).all() and (term.current_morphology == -1).all()
    assert not term.transition_mask.any()
    if not play:
        assert roots[:, 6].std() > .1  # Global yaw is randomized, unlike the old teacher.
    for _ in range(3):
        term._resample_command(torch.arange(n))
        target = term.requested_velocity
        pure = (target.abs() > 1e-6).sum(-1) == 1
        assert not term.standing_mode.any() and (term.current_task == 2).all()
        for axis in range(3):
            for sign in (-1, 1):
                assert (pure & (target[:, axis]*sign > 0)).any()
        assert (target.abs().sum(-1) == 0).any()


@pytest.mark.parametrize('yaw', [0., math.pi/2, math.pi, -math.pi/2])
def test_real_site_frame_tracks_site_velocity_even_when_com_disagrees(monkeypatch, yaw):
    cfg = task.site_snake_env_cfg()
    env = SimpleNamespace(cfg=cfg, num_envs=1, device='cpu', step_dt=.02, common_step_counter=0)
    term = cast(PrivilegedOmniCommandCfg, cfg.commands['transform']).build(env)
    term.command[:, 0] = -1
    term.velocity_command[:] = torch.tensor([.1, 0., 0.])
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    yaw_q = torch.tensor([math.cos(yaw/2), 0., 0., math.sin(yaw/2)])
    pose = quat_mul(yaw_q, torch.tensor([2**-.5, 0., -2**-.5, 0.]))
    velocity = torch.tensor([.1*math.cos(yaw), .1*math.sin(yaw), 0.]).reshape(1, 1, 3)
    data = SimpleNamespace(site_quat_w=pose.reshape(1, 1, 4), site_lin_vel_w=velocity.clone(),
                           site_ang_vel_w=torch.zeros(1, 1, 3), body_com_lin_vel_w=torch.zeros(1, 1, 3))
    env.scene = {'robot': SimpleNamespace(data=data, find_sites=lambda _: ([0], ['foot_0_sole']))}
    monkeypatch.setattr(omni.transform, 'body_masses', lambda env: torch.ones(1, 1))
    torch.testing.assert_close(task.mdp.tracking_velocity(env), torch.tensor([[.1, 0., 0.]]), atol=1e-6, rtol=1e-5)
    assert task.linear_tracking(env).item() == pytest.approx(1.)
    assert task.velocity_error(env, 0).item() == pytest.approx(0., abs=1e-6)
    # Existing omni tasks continue to track COM; the new snake is the sole exception.
    cfg.site_snake_omni = False
    torch.testing.assert_close(task.mdp.tracking_velocity(env), torch.zeros(1, 3))
    cfg.site_snake_omni = True
    data.site_lin_vel_w.zero_()
    data.body_com_lin_vel_w[:] = velocity
    torch.testing.assert_close(task.mdp.tracking_velocity(env), torch.zeros(1, 3))
    assert task.linear_tracking(env).item() < .2



def test_standing_target_rejected_before_any_command_mutation(monkeypatch):
    n = 4
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, common_step_counter=0)
    env._omni_initial_standing = torch.zeros(n, dtype=torch.bool)
    term = task.SiteSnakeCommandCfg(training=False, play_random_velocity=True).build(env)
    monkeypatch.setattr(task.mdp, 'configuration_ready', lambda env, relaxed=False: torch.ones(n, dtype=torch.bool))
    term.reset(torch.arange(n))
    old = {key: value.clone() for key, value in {
        'command': term.command, 'requested': term.requested_velocity,
        'manual': term.manual_control, 'timer': term.time_left,
        'task': term.current_task, 'morphology': term.current_morphology,
    }.items()}
    with pytest.raises(ValueError, match='only accepts the snake target'):
        term.set_target('stand', torch.tensor([1]))
    for key, actual in {
        'command': term.command, 'requested': term.requested_velocity,
        'manual': term.manual_control, 'timer': term.time_left,
        'task': term.current_task, 'morphology': term.current_morphology,
    }.items():
        torch.testing.assert_close(actual, old[key])
    term.set_velocity(.07, -.03, .2, torch.tensor([0]))
    term.stop(torch.tensor([1]))
    term.set_target('snake', torch.tensor([2]))
    torch.testing.assert_close(term.requested_velocity[0], torch.tensor([.07, -.03, .2]))
    torch.testing.assert_close(term.requested_velocity[1], torch.zeros(3))
    torch.testing.assert_close(term.requested_velocity[2:], old['requested'][2:])
    assert (term.current_task == 2).all() and (term.current_morphology == -1).all()
    assert not term.transition_mask.any()
    assert term.manual_control.tolist() == [True, True, True, False]


@pytest.mark.parametrize('mode', ['stand', 'source', 'invalid'])
def test_site_snake_config_rejects_any_other_mode(mode):
    with pytest.raises(ValueError, match="requires mode='snake'"):
        task.SiteSnakeCommandCfg(mode=mode)
    cfg = task.SiteSnakeCommandCfg()
    cfg.mode = mode
    with pytest.raises(ValueError, match="requires mode='snake'"):
        cfg.build(None)


@pytest.mark.parametrize('speed', [.03, .05, .08, .1, .15, .2])
@pytest.mark.parametrize('direction', [(1., 0.), (-1., 0.), (0., 1.), (0., -1.)])
def test_useful_undulating_site_motion_beats_stationary_for_each_translation(monkeypatch, speed, direction):
    x, y = direction
    # Synthetic site samples, not previously measured COM trajectories.
    env = reward_env(monkeypatch, [[speed*x, speed*y, .5], [speed*x, speed*y, -.5], [0., 0., 0.]],
                     [[speed*x, speed*y, 0.]]*3)
    score = 2*task.linear_tracking(env) + task.angular_tracking(env)
    assert score[:2].mean() > score[2]


def test_reward_terms_remain_finite_under_large_tracking_errors(monkeypatch):
    env = reward_env(monkeypatch, [[1e6, -1e6, 1e6]], [[.1, 0., .4]])
    for reward in (task.linear_tracking, task.angular_tracking):
        value = reward(env)
        assert torch.isfinite(value).all() and ((0 <= value) & (value <= 1)).all()


def test_two_tracking_terms_also_handle_stop_and_combined_twists(monkeypatch):
    targets = [[0., 0., 0.], [.1, 0., 0.], [-.1, 0., 0.], [0., .1, 0.],
               [0., -.1, 0.], [0., 0., .4], [0., 0., -.4], [.05, -.05, .3]]
    env = reward_env(monkeypatch, targets, targets)
    cfg = task.site_snake_env_cfg()
    score = sum(term.weight * term.func(env, **term.params) for term in cfg.rewards.values())
    torch.testing.assert_close(score, torch.full((len(targets),), 5.))
    actual = [[x+.03, y+.03, yaw+.3] for x, y, yaw in targets]
    env = reward_env(monkeypatch, actual, targets)
    disturbed = sum(term.weight * term.func(env, **term.params) for term in cfg.rewards.values())
    assert (disturbed < score).all()


def test_linear_width_preserves_low_speed_resolution_and_caps_at_archive_std(monkeypatch):
    target = [[0., 0., 0.], [.08, 0., 0.], [.2, 0., 0.], [.2, .2, 0.]]
    actual = [[.02, 0., 0.], [.14, 0., 0.], [.3, 0., 0.], [.3, .2, 0.]]
    env = reward_env(monkeypatch, actual, target)
    cfg = task.site_snake_env_cfg()
    reward = cfg.rewards['track_linear_velocity']
    torch.testing.assert_close(reward.func(env, **reward.params), torch.full((4,), math.exp(-1)))
    yaw = cfg.rewards['track_angular_velocity']
    assert yaw.params == {'std': .25}


def test_site_command_sampling_uses_archive_scaled_ranges_without_ema_shrink():
    torch.manual_seed(42)
    n = 4096
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, common_step_counter=0)
    term = task.SiteSnakeCommandCfg().build(env)
    term.manual_control[0] = True
    term.requested_velocity[0] = torch.tensor([.13, -.07, .3])
    term._sample_velocity(torch.arange(n))
    command = term.requested_velocity[1:]
    limits = torch.tensor([.2, .2, .5])
    assert (command.abs() <= limits).all()
    assert (command.abs().sum(-1) == 0).any()
    pure = (command.abs() > 1e-6).sum(-1) == 1
    for axis in range(3):
        for sign in (-1, 1):
            magnitude = command[pure & (command[:, axis] * sign > 0), axis].abs()
            assert len(magnitude) > 20
            assert (magnitude >= .4 * limits[axis]).all()
    torch.testing.assert_close(term.requested_velocity[0], torch.tensor([.13, -.07, .3]))
    term.set_velocity(.5, -.5, 2., torch.tensor([0]))
    torch.testing.assert_close(term.requested_velocity[0], limits * torch.tensor([1., -1., 1.]))


@pytest.mark.parametrize('use_target', [False, True])
def test_limit_cost_is_zero_inside_soft_band_and_increases_symmetrically(use_target):
    cfg = task.site_snake_env_cfg()
    limit = math.pi/3
    angles = torch.tensor([0., .8, .9, .95, 1., -1., 1.1]) * limit
    position = angles[:, None].expand(-1, 6).clone()
    data = SimpleNamespace(joint_pos=torch.zeros_like(position) if use_target else position)
    action = SimpleNamespace(commanded_target=position if use_target else torch.zeros_like(position))
    env = SimpleNamespace(cfg=cfg, scene={'robot': SimpleNamespace(data=data)},
                          action_manager=SimpleNamespace(get_term=lambda _: action))
    cost = task.joint_limit_penalty(env, use_target=use_target)
    expected = torch.tensor([0., 0., 0., .05, .1, .1, .2]) * limit * 6
    torch.testing.assert_close(cost, expected, atol=1e-6, rtol=1e-5)


def test_pulling_target_back_from_hard_limit_reduces_total_penalty_before_robot_moves():
    cfg = task.site_snake_env_cfg()
    position = torch.full((2, 6), math.pi/3)
    target = position.clone()
    target[1] = 0.
    action = SimpleNamespace(commanded_target=target)
    env = SimpleNamespace(cfg=cfg, scene={'robot': SimpleNamespace(data=SimpleNamespace(joint_pos=position))},
                          action_manager=SimpleNamespace(get_term=lambda _: action))
    actual = cfg.rewards['joint_pos_limits']
    target_term = cfg.rewards['joint_target_limits']
    score = actual.weight * actual.func(env, **actual.params) + target_term.weight * target_term.func(env, **target_term.params)
    assert score[0].item() == pytest.approx(-2*6*.1*math.pi/3)
    assert score[1] > score[0]
    torch.testing.assert_close(actual.func(env, **actual.params)[0], actual.func(env, **actual.params)[1])


@pytest.mark.parametrize('yaw', [0., math.pi/2, math.pi, -math.pi/2])
def test_site_gravity_tilt_penalty_and_strict_60_degree_reset_are_yaw_invariant(yaw):
    degrees = torch.tensor([0., 30., 59.9, 60., 60.001, 90., 180.], dtype=torch.float64)
    angles = torch.deg2rad(degrees)
    rotation = torch.zeros(len(angles), 4, dtype=torch.float64)
    rotation[:, 0] = (angles/2).cos()
    rotation[:, 2] = (angles/2).sin()
    canonical = rotation.new_tensor([2**-.5, 0., -2**-.5, 0.]).expand_as(rotation)
    yaw_rotation = rotation.new_tensor([math.cos(yaw/2), 0., 0., math.sin(yaw/2)]).expand_as(rotation)
    quat = quat_mul(yaw_rotation, quat_mul(rotation, canonical))
    cfg = task.site_snake_env_cfg()
    data = SimpleNamespace(site_quat_w=quat[:, None], body_link_quat_w=torch.full_like(quat[:, None], float('nan')))
    env = SimpleNamespace(cfg=cfg, scene={'robot': SimpleNamespace(data=data)}, _omni_motion_site_index=0)
    torch.testing.assert_close(task.site_tilt(env), angles, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(task.site_tilt_penalty(env), 1-angles.cos())
    assert task.site_tilt_limit_reached(env).tolist() == [False, False, False, False, True, True, True]
    reward = cfg.rewards['site_tilt']
    assert reward.weight == -1.
    torch.testing.assert_close(reward.weight * reward.func(env), -(1-angles.cos()))
    cfg.sim.mujoco.gravity = (0., 0., -1.)
    torch.testing.assert_close(task.site_tilt(env), angles, atol=1e-7, rtol=1e-5)
