"""Physical origin approach speed and relative axis-angle error reduction rate."""
import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul

from zbot_rl_mjlab.tasks.reconfiguration.basic import mdp


def rotation(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm().clamp_min(1e-15)
    return torch.cat((torch.tensor([math.cos(angle / 2)], dtype=torch.float64), axis * math.sin(angle / 2)))[None]


@pytest.fixture
def sample(monkeypatch):
    g = {
        'delta': torch.tensor([[.4, 0., 0.]], dtype=torch.float64),
        'distance': torch.tensor([.4], dtype=torch.float64),
        'dv': torch.zeros(1, 3, dtype=torch.float64),
        'q0': rotation((0., 1., 0.), 0.),
        'relative_quat': rotation((1., 0., 0.), .8),
        'dw': torch.zeros(1, 3, dtype=torch.float64),
    }
    monkeypatch.setattr(mdp, 'state', lambda env: SimpleNamespace(geometry=lambda: g))
    return SimpleNamespace(step_dt=.02, common_step_counter=1), g


@pytest.mark.parametrize(('dv', 'expected'), [
    ((-.2, 0., 0.), .2), ((.3, 0., 0.), -.3),
    ((0., 3., 0.), 0.), ((-.2, 3., 0.), .2), ((0., 0., 0.), 0.),
])
def test_origin_velocity_projects_only_toward_each_other(sample, dv, expected):
    env, g = sample
    g['dv'][:] = torch.tensor(dv)
    assert mdp.distance_closing_speed(env).item() == pytest.approx(expected)
    assert mdp.distance_velocity_reward(env).item() == pytest.approx(expected)


@pytest.mark.parametrize('distance', [.02, .2, 2.])
def test_approach_speed_is_independent_of_distance_reward_and_control_dt(sample, distance):
    env, g = sample
    g['delta'][:, 0] = distance
    g['distance'][:] = distance
    g['dv'][:, 0] = -.25
    for dt in (.01, .02, .04):
        env.step_dt = dt
        assert mdp.distance_closing_speed(env).item() == pytest.approx(.25)


@pytest.mark.parametrize('angle', [.01, .4, 1.5, 3.])
def test_axis_angle_rate_is_rad_per_second_not_normal_score_change(sample, angle):
    env, g = sample
    g['relative_quat'] = rotation((1., 0., 0.), angle)
    g['dw'][:] = torch.tensor([[-.4, 8., 3.]])
    assert mdp.alignment_closing_rate(env).item() == pytest.approx(.4)
    assert mdp.alignment_velocity_reward(env).item() == pytest.approx(.4)
    g['dw'][:, 0] = .6
    assert mdp.alignment_closing_rate(env).item() == pytest.approx(-.6)
    assert mdp.alignment_velocity_reward(env).item() == pytest.approx(-.6)


def test_projection_matches_finite_difference_of_full_relative_axis_angle(sample):
    env, g = sample
    q0 = rotation((.1, .7, -.2), .9)
    relative = rotation((1., 2., -.5), 1.2)
    tail = quat_mul(q0, relative)
    w0 = torch.tensor([[.3, -.2, .1]], dtype=torch.float64)
    w1 = torch.tensor([[.1, .6, -.5]], dtype=torch.float64)
    g.update(q0=q0, relative_quat=relative, dw=w1-w0)
    eps = 1e-6

    def advance(q, w):
        return quat_mul(rotation(w[0], w.norm().item()*eps), q)

    relative_next = quat_mul(quat_conjugate(advance(q0, w0)), advance(tail, w1))
    angle = lambda q: 2 * torch.atan2(q[:, 1:].norm(dim=-1), q[:, 0].abs())
    numerical = -(angle(relative_next)-angle(relative))/eps
    torch.testing.assert_close(mdp.alignment_closing_rate(env), numerical, atol=1e-6, rtol=1e-5)


def test_world_rotation_and_quaternion_sign_do_not_change_rates(sample):
    env, g = sample
    g['dv'][:, 0] = -.3
    g['dw'][:, 0] = -.4
    before = torch.stack((mdp.distance_closing_speed(env), mdp.alignment_closing_rate(env)))
    world = rotation((.3, .5, .1), 1.1)
    for key in ('delta', 'dv', 'dw'):
        g[key] = quat_apply(world, g[key])
    g['q0'] = quat_mul(world, g['q0'])
    g['relative_quat'] = -g['relative_quat']
    after = torch.stack((mdp.distance_closing_speed(env), mdp.alignment_closing_rate(env)))
    torch.testing.assert_close(after, before)


def test_current_motion_is_read_without_history_or_reset_jump(sample):
    env, g = sample
    g['dv'][:, 0] = -.2
    assert mdp.distance_closing_speed(env).item() == pytest.approx(.2)
    g['delta'][:, 0] = .66
    g['distance'][:] = .66
    g['dv'].zero_()
    g['dw'].zero_()
    assert mdp.distance_closing_speed(env).item() == 0.
    assert mdp.alignment_closing_rate(env).item() == 0.
    assert not hasattr(env, '_docking_reward_rates')


@pytest.mark.parametrize('angle', [0., math.pi])
def test_undefined_axis_or_shortest_angle_cusp_is_neutral(sample, angle):
    env, g = sample
    g['relative_quat'] = rotation((1., 0., 0.), angle)
    g['dw'][:, 0] = 2.
    assert mdp.alignment_closing_rate(env).item() == 0.
    g['distance'].zero_()
    g['delta'].zero_()
    g['dv'][:, 0] = 2.
    assert mdp.distance_closing_speed(env).item() == 0.
