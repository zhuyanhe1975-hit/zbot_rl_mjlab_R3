"""Student frames must not acquire privileged fields by teacher masking."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.tasks.universal import student_observations as observations


def environment():
    n = 4
    data = SimpleNamespace(
        joint_pos=torch.arange(24).reshape(n, 6).float() / 100,
        joint_pos_biased=torch.arange(24).reshape(n, 6).float() / 100 + .015,
        joint_vel=torch.full((n, 6), .2),
        body_link_quat_w=torch.tensor([1., 0., 0., 0.]).expand(n, 2, 4).clone(),
        body_link_ang_vel_w=torch.arange(24).reshape(n, 2, 3).float(),
    )
    robot = SimpleNamespace(data=data, find_bodies=lambda _: ([0], ['base']))
    command = SimpleNamespace(
        current_morphology=torch.tensor([1., 0., -1., 0.]),
        current_task=torch.arange(4), straightening_mask=torch.tensor([False, False, False, True]),
        velocity_command=torch.full((n, 3), .1), requested_velocity=torch.full((n, 3), .2),
        gait_frequency=torch.full((n,), .7), transition_mask=torch.tensor([False, True, False, True]),
    )
    action = SimpleNamespace(commanded_target=torch.full((n, 6), .3), raw_action=torch.full((n, 6), .4))
    env = SimpleNamespace(
        scene={'robot': robot}, command_manager=SimpleNamespace(get_term=lambda _: command),
        action_manager=SimpleNamespace(get_term=lambda _: action),
        cfg=SimpleNamespace(student_observation_noise=False, play_transition_sync=False),
        episode_length_buf=torch.arange(n), step_dt=.02,
    )
    return env, data, command, action


def test_sensor_and_software_layout():
    env, data, command, action = environment()
    state = observations.student_sensor_state(env)
    assert state.shape == (4, 56)
    torch.testing.assert_close(state[:, :6], data.joint_pos_biased)
    assert torch.all(state[:, :6] != data.joint_pos)
    torch.testing.assert_close(state[:, 6:12], data.joint_vel)
    torch.testing.assert_close(state[:, 12:18], action.commanded_target)
    torch.testing.assert_close(state[:, 18:24], action.raw_action)
    torch.testing.assert_close(state[:, 28:31], torch.tensor([0., 0., -1.]).expand(4, 3))
    torch.testing.assert_close(state[:, 31:34], data.body_link_ang_vel_w[:, 0])
    torch.testing.assert_close(state[:, 38], command.current_morphology)
    torch.testing.assert_close(state[:, 39], torch.tensor([0., 1., 2., 3.25]))
    torch.testing.assert_close(state[:, 40:43], command.velocity_command)
    torch.testing.assert_close(state[:, 53:56], command.requested_velocity)
    forbidden = [*range(24, 28), *range(34, 38), *range(46, 53)]
    assert torch.count_nonzero(state[:, forbidden]) == 0


def test_privileged_values_and_snake_foot_imu_do_not_leak(monkeypatch):
    from zbot_rl_mjlab.tasks.universal import guided_observations, privileged

    def forbidden(_):
        pytest.fail('Student must not construct and mask a privileged observation')

    monkeypatch.setattr(guided_observations, 'local_reference_privileged_state', forbidden)
    monkeypatch.setattr(privileged, 'privileged_state', forbidden)
    env, data, _, _ = environment()
    before = observations.student_sensor_state(env)
    data.joint_pos = torch.randn(4, 6) * 100
    data.body_link_quat_w[:, 1] = torch.randn(4, 4)
    data.body_link_ang_vel_w[:, 1] = torch.randn(4, 3) * 100
    data.body_link_pos_w = torch.randn(4, 2, 3) * 100
    data.body_link_lin_vel_w = torch.randn(4, 2, 3) * 100
    data.site_quat_w = torch.randn(4, 2, 4)
    data.site_pos_w = torch.randn(4, 2, 3)
    torch.testing.assert_close(observations.student_sensor_state(env), before)


def test_base_imu_rotates_in_base_coordinates_for_snake_too():
    env, data, _, _ = environment()
    data.body_link_quat_w[:] = torch.tensor([2 ** -.5, 2 ** -.5, 0., 0.])
    actual = observations.student_sensor_state(env)
    torch.testing.assert_close(actual[:, 28:31], torch.tensor([0., -1., 0.]).expand(4, 3), atol=1e-6, rtol=0)


def test_noise_changes_only_sensor_slots_and_is_not_in_place():
    env, data, _, _ = environment()
    baseline = observations.student_sensor_state(env)
    positions = data.joint_pos.clone()
    encoders = data.joint_pos_biased.clone()
    env.cfg.student_observation_noise = True
    torch.manual_seed(7)
    noisy = observations.student_sensor_state(env)
    sensor = torch.zeros(56, dtype=torch.bool)
    sensor[:12] = True
    sensor[28:34] = True
    assert torch.all(noisy[:, sensor] != baseline[:, sensor])
    torch.testing.assert_close(noisy[:, ~sensor], baseline[:, ~sensor])
    torch.testing.assert_close(data.joint_pos, positions)
    torch.testing.assert_close(data.joint_pos_biased, encoders)


def test_play_transition_gates_requests_without_mutating_command():
    env, _, command, _ = environment()
    env.cfg.play_transition_sync = True
    state = observations.student_sensor_state(env)
    assert torch.count_nonzero(state[command.transition_mask, 40:43]) == 0
    assert torch.count_nonzero(state[command.transition_mask, 53:56]) == 0
    assert torch.all(command.requested_velocity == .2)


def test_default_history_and_noise_contract():
    defaults = {field.name: field.default for field in fields(observations.GuidedStudentEnvCfg)}
    assert defaults['student_history_length'] == 5
    assert defaults['student_observation_noise'] is True
