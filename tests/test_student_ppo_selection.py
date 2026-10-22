"""Validation selection must retain weak skills and reject malformed metrics."""

import json

import pytest

from zbot_rl_mjlab.student_ppo_selection import (
    REQUIRED_METRICS,
    StudentPPOCheckpointSelector,
)


def metrics(**overrides):
    return {**dict.fromkeys(REQUIRED_METRICS, .5), **overrides}


@pytest.mark.parametrize('failed_skill', ['biped_backward_success', 'snake_backward_success',
                                          'biped_mixed_negative_success', 'ascent_handoff_fraction'])
def test_forward_improvement_cannot_hide_signed_or_handoff_regression(failed_skill):
    selector = StudentPPOCheckpointSelector(metrics())
    values = metrics(biped_forward_success=1., biped_moving_success=.9)
    values[failed_skill] = 0.
    result = selector.observe(values)
    assert not result['eligible'] and not result['improved']
    assert failed_skill in result['degraded_metrics']
    assert selector.best_metrics == metrics()


@pytest.mark.parametrize('failed_skill', ['ascent_success', 'descent_success',
                                          'biped_stop_success', 'snake_stop_success'])
def test_handoff_improvement_cannot_hide_failed_settling_or_stop(failed_skill):
    selector = StudentPPOCheckpointSelector(metrics())
    values = metrics(ascent_handoff_fraction=1., descent_handoff_fraction=1.)
    values[failed_skill] = 0.
    result = selector.observe(values)
    assert result['score'] == [.5, .75]
    assert not result['eligible'] and not result['improved']
    assert failed_skill in result['degraded_metrics']
    assert selector.best_metrics == metrics()


def test_patience_counts_consecutive_failures_and_resets_on_eligible_validation():
    selector = StudentPPOCheckpointSelector(metrics())
    bad = metrics(ascent_handoff_fraction=0.)
    for _ in range(2):
        assert not selector.observe(bad)['should_stop']
    assert selector.observe(metrics())['consecutive_failures'] == 0
    for _ in range(2):
        assert not selector.observe(bad)['should_stop']
    assert selector.observe(bad)['should_stop']


def test_baseline_is_best_and_bottleneck_precedes_mean_with_mean_breaking_ties():
    selector = StudentPPOCheckpointSelector(metrics(snake_moving_success=0.))
    assert not selector.observe(metrics(snake_moving_success=0.))['improved']
    tied = metrics(snake_moving_success=0., biped_moving_success=.75)
    assert selector.observe(tied)['improved']
    assert selector.best_score == (0., .4375)
    lifted = metrics(snake_moving_success=.1, biped_moving_success=.25,
                     ascent_handoff_fraction=.25, descent_handoff_fraction=.25)
    assert selector.observe(lifted)['improved']
    assert selector.best_score == (.1, .2125)
    assert not selector.observe(tied)['improved']


def test_tolerance_is_absolute_and_boundary_is_eligible():
    selector = StudentPPOCheckpointSelector(metrics(biped_backward_success=.75))
    assert selector.observe(metrics())['eligible']
    assert not selector.observe(metrics(biped_backward_success=.49))['eligible']


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -.1, 1.1, None])
def test_invalid_metrics_never_improve_and_count_toward_stop(invalid):
    selector = StudentPPOCheckpointSelector(metrics(), patience=1)
    result = selector.observe(metrics(snake_moving_success=invalid))
    assert not result['eligible'] and not result['improved']
    assert result['should_stop'] and result['score'] is None
    assert selector.best_metrics == metrics()


def test_missing_metric_rejects_candidate_and_invalid_baseline_raises():
    values = metrics()
    del values['snake_backward_success']
    selector = StudentPPOCheckpointSelector(metrics())
    assert not selector.observe(values)['eligible']
    with pytest.raises(ValueError, match='snake_backward_success'):
        StudentPPOCheckpointSelector(values)


def test_json_state_roundtrip_preserves_best_baseline_and_patience():
    selector = StudentPPOCheckpointSelector(metrics())
    assert selector.observe(metrics(biped_moving_success=.75))['improved']
    bad = metrics(ascent_handoff_fraction=0.)
    selector.observe(bad)
    state = json.loads(json.dumps(selector.state_dict(), allow_nan=False))
    restored = StudentPPOCheckpointSelector.from_state_dict(state)
    assert restored.state_dict() == selector.state_dict()
    assert restored.best_score == selector.best_score
    for _ in range(2):
        assert restored.observe(bad) == selector.observe(bad)
    assert restored.consecutive_failures == 3


def test_invalid_restored_best_is_rejected():
    state = StudentPPOCheckpointSelector(metrics()).state_dict()
    state['best_metrics']['ascent_handoff_fraction'] = 0.
    with pytest.raises(ValueError, match='baseline'):
        StudentPPOCheckpointSelector.from_state_dict(state)


@pytest.mark.parametrize('settings', [
    {'tolerance': float('nan')}, {'tolerance': -.1}, {'tolerance': 1.1},
    {'tolerance': True}, {'patience': 0}, {'patience': True}, {'patience': 1.5},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        StudentPPOCheckpointSelector(metrics(), **settings)
