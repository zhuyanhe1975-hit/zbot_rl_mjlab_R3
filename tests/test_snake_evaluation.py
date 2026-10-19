"""蛇形验收不能把静止、均值抵消、复位或失效数据当成已学会运动。"""

import math

import pytest
import torch

from scripts.evaluate_snake import scenarios, summarize


def inputs(target=(.05, 0., 0.)):
    velocity = torch.tensor(target).expand(8, 4, 3).clone()
    tilt = torch.full((8, 4), .1)
    contact = torch.ones(8, 4)
    acceleration = torch.zeros(8, 4)
    failed = torch.zeros(4, dtype=torch.bool)
    collided = torch.zeros_like(failed)
    return [velocity, tilt, contact, acceleration, failed, collided, target]


def test_all_requested_directions_and_combination():
    cases = scenarios(.05, .3)
    assert len(cases) == 8
    assert cases["backward"] == (-.05, 0., 0.)
    assert cases["right"] == (0., -.05, 0.)
    assert cases["turn_right"] == (0., 0., -.3)
    assert cases["forward_turn"] == (.05, 0., .3)


def test_tracking_safe_motion_passes():
    result = summarize(*inputs())
    assert result["success_rate"] == 1.
    assert result["failure_rate"] == 0.


def test_alive_but_stationary_does_not_pass_motion():
    data = inputs()
    data[0].zero_()
    result = summarize(*data)
    assert result["failure_rate"] == 0.
    assert result["success_rate"] == 0.


def test_stopping_checks_per_frame_drift_not_average_cancellation():
    data = inputs((0., 0., 0.))
    data[0][:4, :, 0] = .04
    data[0][4:, :, 0] = -.04
    result = summarize(*data)
    assert result["actual_velocity_m_s_m_s_rad_s"][0] == pytest.approx(0.)
    assert result["success_rate"] == 0.


def test_latched_failure_and_collision_cannot_disappear_after_reset():
    data = inputs()
    data[4][0] = True
    data[5][1] = True
    result = summarize(*data)
    assert result["per_env_success"] == [False, False, True, True]
    assert result["failure_rate"] == .25
    assert result["self_collision_rate"] == .25


def test_camera_and_contact_peak_checks_include_preparation():
    data = inputs()
    result = summarize(*data, trial_tilt_peak=torch.tensor([.6, .1, .1, .1]),
                       trial_contact_peak=torch.tensor([1., 4., 1., 1.]))
    assert result["per_env_success"] == [False, False, True, True]
    assert result["head_tilt_peak_deg"] == pytest.approx(math.degrees(.6))
    assert result["contact_peak_bodyweights"] == 4.
    assert result["motion_success_rate"] == 1.
    assert result["camera_stable_rate"] == .75
    assert result["contact_safe_rate"] == .75


def test_small_head_tilt_does_not_hide_large_acceleration_jitter():
    data = inputs()
    data[3][:, 0] = 2.
    result = summarize(*data)
    assert result["head_tilt_peak_deg"] < 15.
    assert result["per_env_success"] == [False, True, True, True]


def test_nonfinite_telemetry_is_failed_and_json_safe():
    import json

    data = inputs()
    data[0][0, 0, 0] = float("nan")
    result = summarize(*data)
    assert result["failure_rate"] == .25
    assert not result["per_env_success"][0]
    assert result["actual_velocity_m_s_m_s_rad_s"][0] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("target", [(0., 0., .3), (.05, 0., .3), (0., 0., 0.)])
def test_turning_mixed_motion_and_stationary_targets_are_valid(target):
    assert summarize(*inputs(target))["success_rate"] == 1.


def test_rejects_empty_measurement():
    data = inputs()
    data[0] = torch.empty(0, 4, 3)
    with pytest.raises(ValueError, match="非空"):
        summarize(*data)
