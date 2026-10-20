"""Motion demand must stay broad even before the teacher learns to track."""

from pathlib import Path

import pytest
import torch
import yaml

from zbot_rl_mjlab.tasks.universal.teacher_sampling import (
    sample_motion_commands,
    training_roles,
)


def test_mixed_biped_commands_reproduce_successful_archive_ranges():
    path = Path(__file__).resolve().parents[1] / 'bests/omni_velocity_energy_2026-10-10_iter499/env.yaml'
    saved = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    ranges = saved['commands']['target_speed']['ranges']
    torch.manual_seed(812)
    commands = sample_motion_commands(torch.ones(32000, dtype=torch.bool))
    mixed = commands[(commands != 0).sum(dim=1) == 3]
    assert len(mixed) > 15000
    for i, name in enumerate(('lin_vel_x', 'lin_vel_y', 'ang_vel_z')):
        lower, upper = (float(x) for x in ranges[name])
        assert mixed[:, i].min().item() == pytest.approx(lower, abs=.003)
        assert mixed[:, i].max().item() == pytest.approx(upper, abs=.003)
        assert mixed[:, i].mean().item() == pytest.approx((lower + upper) / 2, abs=.005)


@pytest.mark.parametrize('standing', [True, False])
def test_full_skill_coverage_and_nontrivial_signed_axis_demand(standing):
    torch.manual_seed(513)
    commands = sample_motion_commands(torch.full((32000,), standing, dtype=torch.bool))
    nonzero = (commands != 0).sum(dim=1)
    assert (nonzero == 0).float().mean().item() == pytest.approx(1 / 8, abs=.01)
    assert (nonzero == 1).float().mean().item() == pytest.approx(3 / 8, abs=.01)
    assert (nonzero == 3).float().mean().item() == pytest.approx(1 / 2, abs=.01)
    endpoints = ((.5, .3), (.2, .2), (.5, .5)) if standing else ((.25, .25), (.2, .2), (.8, .8))
    for axis, magnitudes in enumerate(endpoints):
        for sign, maximum in zip((1, -1), magnitudes, strict=True):
            values = commands[(nonzero == 1) & (commands[:, axis] * sign > 0), axis].abs()
            assert len(values) / len(commands) == pytest.approx(1 / 16, abs=.008)
            assert values.min() >= .4 * maximum - 1e-7
            assert values.max() <= maximum


def test_stationary_policy_does_not_earn_near_perfect_motion_tracking():
    # There is deliberately no tracking-EMA argument capable of shrinking demand.
    torch.manual_seed(189)
    commands = sample_motion_commands(torch.ones(32000, dtype=torch.bool))
    moving = commands[commands.abs().sum(dim=1) > 0]
    static_reward = (2 * torch.exp(-moving[:, 0].square() / .1**2)
                     + torch.exp(-moving[:, 1].square() / .1**2)
                     + torch.exp(-moving[:, 2].square() / .25**2))
    assert static_reward.mean() < 3.0  # The shrunken v2 requests yielded about 3.81 / 4.


def test_explicit_limits_clamp_ranges_and_axis_amplitudes():
    torch.manual_seed(24)
    limits = (.1, .05, .2)
    values = sample_motion_commands(torch.arange(10000) % 2 == 0, limits)
    assert torch.all(values.abs() <= torch.tensor(limits))
    for i, cap in enumerate(limits):
        assert values[:, i].max().item() == pytest.approx(cap, abs=.001)
        assert values[:, i].min().item() == pytest.approx(-cap, abs=.001)
    assert sample_motion_commands(torch.empty(0, dtype=torch.bool)).shape == (0, 3)


@pytest.mark.parametrize('limits', [(0., .2, .8), (float('nan'), .2, .8), (.2, float('inf'), .8)])
def test_invalid_limits_rejected(limits):
    with pytest.raises(ValueError, match='limits'):
        sample_motion_commands(torch.ones(1, dtype=torch.bool), limits)


def test_training_roles_have_exact_global_quotas_and_partial_reset_consistency():
    ids = torch.arange(1000)
    roles = training_roles(1000, ids)
    assert torch.bincount(roles).tolist() == [400, 400, 100, 100]
    subset = torch.tensor([999, 300, 401, 800, 0, 799, 900])
    assert torch.equal(training_roles(1000, subset), roles[subset])
    assert training_roles(10, torch.arange(10)).tolist() == [0] * 4 + [1] * 4 + [2, 3]


@pytest.mark.parametrize('n', [1, 4, 9])
def test_small_scenes_leave_role_choice_to_existing_sampler(n):
    assert training_roles(n, torch.arange(n)).tolist() == [-1] * n


@pytest.mark.parametrize('fractions', [(.4, .4, .1, .2), (.5, .5, 0., 0.), (.4, .4, float('nan'), .1)])
def test_invalid_role_fractions_rejected(fractions):
    with pytest.raises(ValueError, match='fractions'):
        training_roles(1000, torch.arange(2), fractions)


@pytest.mark.parametrize('ids', [torch.tensor([-1]), torch.tensor([10]), torch.tensor([1.])])
def test_invalid_global_ids_rejected(ids):
    with pytest.raises(ValueError, match='ids'):
        training_roles(10, ids)


@pytest.mark.parametrize('n', [0, -1])
def test_invalid_environment_count_rejected(n):
    with pytest.raises(ValueError, match='num_envs'):
        training_roles(n, torch.empty(0, dtype=torch.long))


@pytest.mark.parametrize('minimum', [0., -1., 1.01, float('nan')])
def test_axis_min_fraction_rejects_invalid_ranges(minimum):
    with pytest.raises(ValueError, match='axis_min_fraction'):
        sample_motion_commands(torch.ones(8, dtype=torch.bool), axis_min_fraction=minimum)


def test_low_speed_axis_commands_cover_the_default_play_request():
    torch.manual_seed(42)
    standing = torch.ones(32768, dtype=torch.bool)
    archived = sample_motion_commands(standing)
    torch.manual_seed(42)
    initial = sample_motion_commands(standing, (.15, .08, .4), axis_min_fraction=.1)
    old_forward_only = (archived[:, 0] > 0) & archived[:, 1:].eq(0).all(-1)
    new_forward_only = (initial[:, 0] > 0) & initial[:, 1:].eq(0).all(-1)
    # Former pure-forward samples start at .2m/s, leaving .05m/s unseen on that axis.
    assert archived[old_forward_only, 0].min() >= .2
    requests = initial[new_forward_only, 0]
    assert requests.max() <= .15
    assert ((requests > .04) & (requests < .06)).sum() > 100
    assert initial[:, 0].min() < -.1
    assert initial[:, 1].min() < -.05
    assert initial[:, 2].min() < -.3
