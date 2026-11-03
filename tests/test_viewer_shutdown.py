"""Closing a window must not wait for a pending overlay acknowledgement."""

from threading import Event
from types import SimpleNamespace

import mujoco
import pytest

from zbot_rl_mjlab.viewer import _AsyncOverlayHandle


@pytest.mark.parametrize('kind', ['set_texts', 'set_figures'])
def test_close_does_not_join_a_stalled_overlay_and_discards_pending_updates(kind):
    entered, release = Event(), Event()
    calls = []
    running = True

    def stalled(value):
        calls.append(value)
        entered.set()
        assert release.wait(5), 'test did not release overlay worker'

    def close():
        nonlocal running
        running = False

    handle = _AsyncOverlayHandle(SimpleNamespace(
        is_running=lambda: running, close=close,
        set_texts=stalled, set_figures=stalled,
    ))
    value = ('first',) if kind == 'set_texts' else []
    try:
        getattr(handle, kind)(value)
        assert entered.wait(2)
        for index in range(100):
            handle.set_texts((index,))
            handle.set_figures([])
        assert len(handle._pending) == 2
        handle.close()
        assert not running
        assert handle._thread.is_alive()
        assert not handle._pending
    finally:
        release.set()
        handle.close()
        handle._thread.join(2)
    assert not handle._thread.is_alive()
    assert len(calls) == 1


def test_figures_are_snapshots_before_worker_uses_them():
    entered, release = Event(), Event()
    received = []

    def texts(value):
        entered.set()
        assert release.wait(5)

    handle = _AsyncOverlayHandle(SimpleNamespace(
        is_running=lambda: True, close=lambda: None,
        set_texts=texts, set_figures=lambda value: received.extend(value),
    ))
    try:
        handle.set_texts(('',))
        assert entered.wait(2)
        rect, figure = mujoco.MjrRect(0, 0, 100, 200), mujoco.MjvFigure()
        figure.linepnt[0] = 7
        handle.set_figures([(rect, figure)])
        snapshot_rect, snapshot_figure = handle._pending['set_figures'][0]
        rect.width = 300
        figure.linepnt[0] = 99
        assert snapshot_rect.width == 100
        assert snapshot_figure.linepnt[0] == 7
    finally:
        handle.close()
        release.set()
        handle._thread.join(2)


def test_native_viewer_setup_and_close_use_the_async_handle(monkeypatch):
    from mjlab.viewer import NativeMujocoViewer

    from zbot_rl_mjlab.viewer import FollowCameraViewer

    closed = Event()
    raw = SimpleNamespace(is_running=lambda: not closed.is_set(), close=closed.set)
    monkeypatch.setattr(NativeMujocoViewer, 'setup', lambda viewer: setattr(viewer, 'viewer', raw))
    viewer = FollowCameraViewer.__new__(FollowCameraViewer)
    viewer.log = lambda *args: None
    viewer.setup()
    wrapped = viewer.viewer
    try:
        assert isinstance(wrapped, _AsyncOverlayHandle)
        assert wrapped._handle is raw
        viewer.close()
        assert viewer.viewer is None
        assert closed.is_set()
    finally:
        wrapped.close()
        wrapped._thread.join(2)
    assert not wrapped._thread.is_alive()
