"""回放相机按实际时间平滑XY跟踪，保留鼠标视角和高度。"""

import math
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest

from zbot_rl_mjlab import viewer as module


def test_smoothed_follow_preserves_height_and_does_not_jump_to_target():
    actual = module.follow_lookat([0., 0., .2], [1., -1., 5.], dt=.1, smoothing_time=.2)
    weight = 1 - math.exp(-.5)
    np.testing.assert_allclose(actual, [weight, -weight, .2])


def test_filter_is_independent_of_render_frame_rate():
    results = []
    for dt, frames in ((.1, 10), (.01, 100)):
        value = np.array([0., 0., .3])
        for _ in range(frames):
            value = module.follow_lookat(value, [1., 2., 9.], dt=dt, smoothing_time=.2)
        results.append(value)
    np.testing.assert_allclose(results[0], results[1])


def test_high_frequency_sway_is_attenuated():
    value = np.zeros(3)
    filtered = []
    for index in range(300):
        target = [.02 if index % 2 else -.02, 0., 0.]
        value = module.follow_lookat(value, target, dt=1 / 60, smoothing_time=.2)
        filtered.append(value[0])
    assert np.std(filtered[100:]) < .002


def test_zero_delta_holds_and_invalid_target_keeps_valid_camera():
    original = [1., 2., .3]
    np.testing.assert_equal(module.follow_lookat(original, [9., 8., 7.], dt=0., smoothing_time=.2), original)
    np.testing.assert_equal(module.follow_lookat(original, [math.nan, 8., 7.], dt=.1, smoothing_time=.2), original)


@pytest.mark.parametrize('bad', [-.1, math.inf, math.nan])
def test_invalid_smoothing_time_rejected(bad):
    with pytest.raises(ValueError, match='smoothing'):
        module.follow_lookat([0., 0., 0.], [1., 0., 0.], dt=.1, smoothing_time=bad)


def test_viewer_uses_smoothed_tracking_and_resets_time_when_toggled(monkeypatch):
    instance = cast(Any, object.__new__(module.FollowCameraViewer))
    camera = cast(Any, mujoco).MjvCamera()
    camera.lookat[:] = [0., 0., .2]
    camera.azimuth, camera.elevation, camera.distance = 42., -25., 1.5
    instance.viewer = SimpleNamespace(cam=camera, is_running=lambda: True, lock=nullcontext)
    instance._follow = True
    instance._follow_last_time = None
    instance.follow_smoothing_time = .2
    instance.frame_time = .1
    instance._follow_point = lambda: np.array([1., -1., 5.])
    clock = iter([10., 10.1, 100.])
    monkeypatch.setattr(module.time, 'perf_counter', lambda: next(clock))
    instance._apply_follow()
    instance._apply_follow()
    np.testing.assert_allclose(camera.lookat, [1 - math.exp(-1), math.exp(-1) - 1, .2])
    assert (camera.azimuth, camera.elevation, camera.distance) == (42., -25., 1.5)
    instance._toggle_follow()
    held = camera.lookat.copy()
    instance._apply_follow()
    np.testing.assert_equal(camera.lookat, held)
    instance._toggle_follow()
    instance._apply_follow()
    assert held[0] < camera.lookat[0] < 1.
    instance.follow_smoothing_time = 0.
    monkeypatch.setattr(module.time, 'perf_counter', lambda: 101.)
    instance._apply_follow()
    np.testing.assert_allclose(camera.lookat, [1., -1., .2])
