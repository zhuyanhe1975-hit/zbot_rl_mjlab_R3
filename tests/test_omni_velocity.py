import torch
from mjlab.tasks.registry import load_env_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.experiments import walking as recipes
from zbot_rl_mjlab.tasks.walking import mdp as omni

TASK = recipes.OMNI_VELOCITY_ENERGY_TASK_ID


def test_omni_tracks_three_commands_without_conflicting_terms():
    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play)
        for group in cfg.observations.values():
            assert "velocity_command" in group.terms
            assert "yaw_error" not in group.terms
            assert "target_speed" not in group.terms
        assert {"track_forward_speed", "track_lateral_speed", "track_yaw_rate"} <= set(
            cfg.rewards
        )
        assert "lateral_velocity" not in cfg.rewards
        assert "yaw_drift" not in cfg.rewards
        ranges = cfg.commands["target_speed"].ranges
        assert ranges.lin_vel_x == recipes.LIN_VEL_X
        assert ranges.lin_vel_y == recipes.LIN_VEL_Y
        assert ranges.ang_vel_z == recipes.ANG_VEL_Z


def test_yaw_rate_reward_uses_counter_clockwise_positive(monkeypatch):
    command = torch.tensor([[0.0, 0.0, 0.5]])
    monkeypatch.setattr(omni, "velocity_command", lambda env: command)
    # base_ang_vel 的第三轴朝下，因此逆时针转动 0.5 rad/s 对应 -0.5。
    monkeypatch.setattr(
        omni, "base_ang_vel", lambda env: torch.tensor([[0.0, 0.0, -0.5]])
    )
    assert torch.allclose(omni.track_yaw_rate(None), torch.ones(1))
    monkeypatch.setattr(
        omni, "base_ang_vel", lambda env: torch.tensor([[0.0, 0.0, 0.5]])
    )
    assert omni.track_yaw_rate(None).item() < 0.01


def test_command_arrows_follow_zbot_forward_left_axes():
    import numpy as np

    from zbot_rl_mjlab.tasks.walking.mdp import command_arrow_segments

    origin = np.array([1.0, 2.0, 0.5])
    forward = np.array([0.0, 1.0, 0.0])  # 机器人偏航 90 度
    left = np.array([-1.0, 0.0, 0.0])
    (lin_a, lin_b), (yaw_a, yaw_b) = command_arrow_segments(
        origin, forward, left, (1.0, 0.0, 0.0), length=0.25
    )
    assert np.allclose(lin_b - lin_a, 0.25 * forward)
    assert np.allclose(yaw_b, yaw_a)
    (lin_a, lin_b), _ = command_arrow_segments(
        origin, forward, left, (0.0, -1.0, 0.0), length=0.25
    )
    assert np.allclose(lin_b - lin_a, -0.25 * left)
    _, (yaw_a, yaw_b) = command_arrow_segments(
        origin, forward, left, (0.0, 0.0, 1.0), length=0.25
    )
    # 从上方看为逆时针：无论偏航角如何，按右手定则均为世界坐标系 +Z。
    assert np.allclose(yaw_a, origin)
    assert np.allclose(yaw_b - yaw_a, [0.0, 0.0, 0.25])
    _, (yaw_a, yaw_b) = command_arrow_segments(
        origin, forward, left, (0.0, 0.0, -0.5), length=0.25
    )
    assert np.allclose(yaw_b - yaw_a, [0.0, 0.0, -0.125])


def test_follow_camera_moves_lookat_in_xy_and_keeps_height():
    import numpy as np

    from zbot_rl_mjlab.viewer import follow_lookat

    moved = follow_lookat([0.0, -0.06, 0.2], [1.5, -2.0, 0.9])
    assert np.allclose(moved, [1.5, -2.0, 0.2])


def test_follow_key_toggles_only_through_the_action_queue():
    from mjlab.viewer.base import ViewerAction

    from zbot_rl_mjlab import viewer as zviewer

    v = object.__new__(zviewer.FollowCameraViewer)
    v._actions = __import__("collections").deque()
    v._follow = False
    v._on_key(ord("~"))  # 未绑定按键：不应向队列添加任何内容。
    assert not v._actions
    v._on_key(zviewer.FOLLOW_KEY)
    action, payload = v._actions.popleft()
    assert action == ViewerAction.CUSTOM
    assert v._handle_custom_action(action, payload) and v._follow
    assert v._handle_custom_action(action, payload) and not v._follow
    assert not v._handle_custom_action(ViewerAction.CUSTOM, "something else")
