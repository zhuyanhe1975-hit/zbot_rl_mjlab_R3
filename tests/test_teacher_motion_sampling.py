"""Motion demand must stay broad even before the teacher learns to track."""

from pathlib import Path
from typing import cast

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


def test_explicit_legacy_stop_probability_preserves_samples_and_rng_state_exactly():
    standing = torch.arange(32000) % 2 == 0
    torch.manual_seed(718)
    default = sample_motion_commands(standing)
    default_state = torch.random.get_rng_state()
    torch.manual_seed(718)
    explicit = sample_motion_commands(standing, stop_probability=1 / 8)
    assert torch.equal(default, explicit)
    assert torch.equal(default_state, torch.random.get_rng_state())


def test_legacy_default_sampling_preserves_known_samples_and_following_random_draw():
    torch.manual_seed(718)
    command = sample_motion_commands(torch.arange(12) % 2 == 0)
    expected = torch.tensor([
        [0., .1920870840549469, 0.],
        [-.0910290777683258, -.011061400175094604, -.515796959400177],
        [-.0022537708282470703, -.007434636354446411, -.1759060025215149],
        [.11841685324907303, 0., 0.],
        [0., 0., 0.],
    ])
    assert torch.equal(command[:5], expected)
    assert torch.equal(torch.rand(3), torch.tensor([.9669855833053589, .6680901646614075, .009715437889099121]))


@pytest.mark.parametrize('standing', [False, True])
@pytest.mark.parametrize('probability', [0., 1.])
def test_stop_probability_endpoints_disable_or_require_stop(standing, probability):
    torch.manual_seed(507)
    command = sample_motion_commands(torch.full((8192,), standing), stop_probability=probability)
    stopped = command.eq(0).all(-1)
    if probability == 1.:
        assert stopped.all()
    else:
        assert not stopped.any()


@pytest.mark.parametrize('standing', [False, True])
def test_one_percent_stops_preserve_relative_axis_and_mixed_coverage(standing):
    torch.manual_seed(137)
    command = sample_motion_commands(torch.full((32000,), standing), stop_probability=.01)
    nonzero = command.ne(0).sum(-1)
    assert nonzero.eq(0).float().mean().item() == pytest.approx(.01, abs=.002)
    moving = nonzero > 0
    assert nonzero[moving].eq(1).float().mean().item() == pytest.approx(3/7, abs=.015)
    assert nonzero[moving].eq(3).float().mean().item() == pytest.approx(4/7, abs=.015)
    for axis in range(3):
        for sign in (-1, 1):
            count = ((nonzero == 1) & (command[:, axis] * sign > 0)).sum().item()
            assert count / moving.sum().item() == pytest.approx(1/14, abs=.008)


@pytest.mark.parametrize('probability', [-.01, 1.01, float('nan'), float('inf'), -float('inf')])
def test_invalid_stop_probability_rejected(probability):
    with pytest.raises(ValueError, match='stop_probability'):
        sample_motion_commands(torch.ones(8, dtype=torch.bool), stop_probability=probability)


def test_legacy_privileged_config_retains_one_eighth_stop_default():
    from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg

    assert PrivilegedOmniCommandCfg(resampling_time_range=(5., 9.)).motion_stop_probability == 1/8


@pytest.mark.parametrize('probability', [-.01, 1.01, float('nan'), float('inf'), -float('inf')])
def test_privileged_config_rejects_invalid_stop_probability(probability):
    from types import SimpleNamespace

    from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg

    with pytest.raises(ValueError, match='motion_stop_probability'):
        PrivilegedOmniCommandCfg(
            resampling_time_range=(5., 9.), motion_stop_probability=probability,
        ).build(SimpleNamespace())


@pytest.mark.parametrize('standing', [False, True])
def test_x_only_bias_preserves_stops_and_other_motion_coverage(standing):
    torch.manual_seed(509)
    command = sample_motion_commands(torch.full((64000,), standing), stop_probability=.01, x_only_probability=.5)
    active = command.ne(0)
    count = active.sum(-1)
    pure_x = active[:, 0] & (count == 1)
    assert pure_x.float().mean().item() == pytest.approx(.5, abs=.012)
    assert count.eq(0).float().mean().item() == pytest.approx(.01, abs=.002)
    for sign in (-1, 1):
        assert (pure_x & (command[:, 0]*sign > 0)).float().mean().item() == pytest.approx(.25, abs=.01)
    for axis in (1, 2):
        assert (active[:, axis] & (count == 1)).float().mean().item() == pytest.approx(.49/6, abs=.008)
    assert count.eq(3).float().mean().item() == pytest.approx(.49*2/3, abs=.012)


@pytest.mark.parametrize('probability', [0., 1.])
def test_x_only_probability_endpoints(probability):
    command = sample_motion_commands(torch.zeros(8192, dtype=torch.bool), stop_probability=0., x_only_probability=probability)
    pure_x = command[:, 0].ne(0) & command[:, 1:].eq(0).all(-1)
    assert pure_x.all() if probability == 1. else not pure_x.any()


@pytest.mark.parametrize('probability', [-.01, 1., float('nan'), float('inf')])
def test_invalid_x_only_probability_or_joint_budget_rejected(probability):
    with pytest.raises(ValueError, match='x_only_probability'):
        sample_motion_commands(torch.zeros(8, dtype=torch.bool), stop_probability=.01, x_only_probability=probability)


def test_archived_site_card_biases_training_without_changing_play_sampling():
    from mjlab.tasks.registry import load_env_cfg

    from zbot_rl_mjlab.archived_site_snake import ArchivedSiteSnakeCommandCfg

    task = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'
    train = cast(ArchivedSiteSnakeCommandCfg, load_env_cfg(task).commands['transform'])
    play = cast(ArchivedSiteSnakeCommandCfg, load_env_cfg(task, play=True).commands['transform'])
    assert train.motion_x_only_probability == .5
    assert play.motion_x_only_probability is None


def test_command_builder_rejects_invalid_x_only_probability():
    from types import SimpleNamespace

    from zbot_rl_mjlab.archived_site_snake import ArchivedSiteSnakeCommandCfg

    with pytest.raises(ValueError, match='motion_x_only_probability'):
        ArchivedSiteSnakeCommandCfg(motion_stop_probability=.01, motion_x_only_probability=1.).build(SimpleNamespace())
