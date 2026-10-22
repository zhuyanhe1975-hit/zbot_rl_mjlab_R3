"""Student rewards distinguish tracking errors and respect skill/event gates."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg
from test_flat_foot import FakeEnv
from test_task_contracts import canonical

import zbot_rl_mjlab
from zbot_rl_mjlab.experiments.omni_teacher import SnakeLegOmniGuidedTeacher
from zbot_rl_mjlab.experiments.student import UniversalOmniStudent
from zbot_rl_mjlab.tasks.universal import guided_rewards, teacher_rewards
from zbot_rl_mjlab.tasks.universal import student_rewards as rewards
from zbot_rl_mjlab.tasks.universal.teacher_rewards import (
    teacher_action_rate,
    teacher_velocity_tracking,
)


def score(actual, command, **kwargs):
    return rewards.student_snake_motion_score(
        torch.tensor(actual, dtype=torch.float32).reshape(-1, 3),
        torch.tensor(command, dtype=torch.float32).reshape(-1, 3),
        **kwargs,
    )


@pytest.mark.parametrize('command', [(.08, 0., 0.), (0., .05, 0.), (0., 0., .3), (.08, .05, .3), (.2, .2, 0.)])
def test_exact_tracking_beats_stopping_and_overspeed_in_both_directions(command):
    desired = torch.tensor([command])
    exact = rewards.student_snake_motion_score(desired, desired)
    torch.testing.assert_close(exact, torch.tensor([4.]))
    for multiplier in (0., .5, 2., -1.):
        actual = desired * multiplier
        result = rewards.student_snake_motion_score(actual, desired)
        assert 0. <= result.item() < exact.item()
        torch.testing.assert_close(result, rewards.student_snake_motion_score(-actual, -desired))


@pytest.mark.parametrize(('command', 'drift'), [
    ((.08, 0., 0.), (.08, 0., .5)),
    ((.08, 0., 0.), (.08, .05, 0.)),
    ((0., 0., .3), (.08, .05, .3)),
    ((0., 0., 0.), (.08, .05, .3)),
])
def test_inactive_axes_and_zero_command_do_not_reward_drift(command, drift):
    assert score(drift, command).item() < score(command, command).item()


def test_combined_command_requires_translation_and_turning_together():
    command = (.08, .05, .3)
    full = score(command, command).item()
    yaw_only = score((0., 0., .3), command).item()
    translation_only = score((.08, .05, 0.), command).item()
    assert max(yaw_only, translation_only) < full / 2.
    # The old .2 m/s magnitude cap rewarded this slower diagonal equally.
    assert score((.1414, .1414, 0.), (.2, .2, 0.)).item() < 4.


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('field', ['actual', 'command'])
def test_nonfinite_rows_receive_zero_without_affecting_valid_rows(bad, field):
    actual = torch.tensor([[.08, .05, .3], [.08, .05, .3]])
    command = actual.clone()
    (actual if field == 'actual' else command)[0, 1] = bad
    torch.testing.assert_close(rewards.student_snake_motion_score(actual, command), torch.tensor([0., 4.]))


@pytest.mark.parametrize('std', [(0., .05, .15), (-.05, .05, .15), (.05, float('nan'), .15), (.05, .05, float('inf')), (.05, .05), (.05, .05, .15, .2)])
def test_tracking_widths_must_be_a_positive_finite_triplet(std):
    with pytest.raises(ValueError):
        score((0., 0., 0.), (0., 0., 0.), std=std)


def test_velocity_tracking_preserves_biped_branch_and_masks_both_transforms(monkeypatch):
    term = SimpleNamespace(
        standing_mode=torch.tensor([True, False, True, False]),
        transition_mask=torch.tensor([False, False, True, True]),
        velocity_command=torch.tensor([[.08, .05, .3]] * 4),
        cfg=SimpleNamespace(velocity_limits=(.25, .2, .8)),
    )
    actual = torch.tensor([[.06, .04, .2], [.08, .05, .3], [.08, .05, .3], [.08, .05, .3]])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term))
    monkeypatch.setattr(rewards.mdp, 'tracking_velocity', lambda _: actual)
    archived = teacher_velocity_tracking(env, std=(.2, .1, .4), axis_weights=(3., 2., 1.))
    value = rewards.student_velocity_tracking(env, std=(.2, .1, .4), axis_weights=(3., 2., 1.))
    torch.testing.assert_close(value, torch.tensor([archived[0].item(), 4., 0., 0.]))
    # The configured snake widths must affect only snake tracking.
    actual[1, 0] += .04
    narrow = rewards.student_velocity_tracking(env, snake_std=(.02, .05, .15))
    broad = rewards.student_velocity_tracking(env, snake_std=(.1, .05, .15))
    assert narrow[1] < broad[1]
    torch.testing.assert_close(narrow[0], broad[0])


def test_touchdown_penalty_is_gated_to_stable_walking(monkeypatch):
    term = SimpleNamespace(
        standing_mode=torch.tensor([True, False, True, False]),
        transition_mask=torch.tensor([False, False, True, True]),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term))
    def touchdown(env, sigma, sensor_name):
        assert sigma == .03 and sensor_name == 'custom_sensor'
        return torch.tensor([1., 2., 3., 4.])
    monkeypatch.setattr(rewards.walking, 'touchdown_flat_penalty', touchdown)
    torch.testing.assert_close(rewards.student_touchdown_flat(env, sigma=.03, sensor_name='custom_sensor'), torch.tensor([1., 0., 0., 0.]))


def test_touchdown_penalty_requires_a_first_contact_event(monkeypatch):
    env = FakeEnv(10., 0., [1, 0], [0., .1], [False, False])
    term = SimpleNamespace(standing_mode=torch.tensor([True]), transition_mask=torch.tensor([False]))
    monkeypatch.setattr(env, 'command_manager', SimpleNamespace(get_term=lambda _: term), raising=False)
    assert rewards.student_touchdown_flat(env).item() == 0.
    env.scene['feet_ground_contact'].compute_first_contact = lambda dt: torch.tensor([[True, False]])
    assert rewards.student_touchdown_flat(env).item() > .9


@pytest.mark.parametrize('play', [False, True])
def test_only_student_train_play_recipe_enables_corrections(play):
    cfg = load_env_cfg(UniversalOmniStudent.task_id, play=play)
    tracking = cfg.rewards['velocity_tracking']
    assert tracking.func is rewards.student_velocity_tracking
    assert tracking.params['snake_std'] == (.05, .05, .15)
    touchdown = cfg.rewards['foot_flat_touchdown']
    assert touchdown.func is rewards.student_touchdown_flat
    assert touchdown.weight == -.2
    assert touchdown.params == {'sigma': .05, 'sensor_name': 'feet_ground_contact'}
    assert cfg.rewards['action_rate'].func is teacher_action_rate
    teacher = load_env_cfg(SnakeLegOmniGuidedTeacher.task_id, play=play)
    assert teacher.rewards['velocity_tracking'].func is teacher_velocity_tracking
    assert 'snake_std' not in teacher.rewards['velocity_tracking'].params
    assert 'foot_flat_touchdown' not in teacher.rewards
    assert teacher.rewards['action_rate'].func is guided_rewards.guided_action_rate


def test_student_action_smoothing_includes_transforms_but_teacher_keeps_archive_gate(monkeypatch):
    term = SimpleNamespace(
        standing_mode=torch.tensor([True, False, True, False]),
        transition_mask=torch.tensor([False, False, True, True]),
    )
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term))
    raw_cost = torch.tensor([1., 2., 3., 4.])
    monkeypatch.setattr(guided_rewards, 'teacher_action_rate', lambda _: raw_cost)
    student = load_env_cfg(UniversalOmniStudent.task_id)
    teacher = load_env_cfg(SnakeLegOmniGuidedTeacher.task_id)
    # Exercise the configured student callback rather than a replacement wrapper.
    monkeypatch.setattr(teacher_rewards.walking, 'action_rate_l2', lambda _: raw_cost)
    torch.testing.assert_close(student.rewards['action_rate'].func(env), torch.tensor([5., 2., 3., 4.]))
    torch.testing.assert_close(teacher.rewards['action_rate'].func(env), torch.tensor([1., 2., 0., 0.]))


@pytest.mark.parametrize('play', [False, True])
def test_building_student_preserves_all_other_experiments_reward_callbacks(play):
    others = [experiment for experiment in zbot_rl_mjlab.EXPERIMENTS
              if not issubclass(experiment, UniversalOmniStudent)]
    before = {experiment.task_id: canonical(load_env_cfg(experiment.task_id, play=play).rewards)
              for experiment in others}
    load_env_cfg(UniversalOmniStudent.task_id, play=play)
    for experiment in others:
        cfg = load_env_cfg(experiment.task_id, play=play)
        assert canonical(cfg.rewards) == before[experiment.task_id]
        for reward in cfg.rewards.values():
            assert reward.func not in (rewards.student_touchdown_flat, rewards.student_velocity_tracking)
