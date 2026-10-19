"""蛇形指令的方向覆盖、平滑和手动锁定契约。"""

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.commands import SnakeVelocityCommandCfg
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


def command_env(num_envs=8, training=False):
    env = SimpleNamespace(num_envs=num_envs, device="cpu", step_dt=.02)
    term = SnakeVelocityCommandCfg(
        resampling_time_range=(4., 8.), training=training,
    ).build(cast(ManagerBasedRlEnv, env))
    return env, term


def test_training_exposes_all_six_directions_from_first_sampling():
    torch.manual_seed(2026)
    _, term = command_env(24000, training=True)
    term.reset(None)
    requested = term.requested_velocity
    nonzero = requested.ne(0).sum(-1)
    assert .82 < nonzero.eq(1).float().mean() < .88
    assert .08 < nonzero.eq(0).float().mean() < .12
    assert .03 < nonzero.eq(3).float().mean() < .07
    for axis in range(3):
        for sign in (-1, 1):
            count = ((nonzero == 1) & (requested[:, axis] * sign > 0)).sum()
            assert count > (4000 if axis < 2 else 900)
    assert requested.abs().le(requested.new_tensor(term.cfg.velocity_limits)).all()
    for axis, limit in enumerate((.25, .25, math.pi / 2)):
        assert requested[:, axis].max() > .95 * limit
        assert requested[:, axis].min() < -.95 * limit


def test_play_starts_stopped_and_keyboard_locks_without_state_reset():
    env, term = command_env()
    env.joints = torch.randn(8, 6)
    env.targets = torch.randn(8, 6)
    joints, targets = env.joints.clone(), env.targets.clone()
    term.reset(None)
    assert term.command.eq(0).all() and term.requested_velocity.eq(0).all()
    term.set_velocity(.05, -.025, .2)
    assert term.command.eq(0).all()
    assert term.manual_control.all() and term.time_left.isinf().all()
    term._resample_command(torch.arange(8))
    assert term.requested_velocity[0].tolist() == pytest.approx([.05, -.025, .2])
    term.compute(.02)
    assert term.command[0, 0] > 0 and term.command[0, 0] < .05
    for _ in range(150):
        term.compute(.02)
    torch.testing.assert_close(term.command, term.requested_velocity, atol=2e-5, rtol=0)
    term.stop()
    assert term.requested_velocity.eq(0).all() and term.command.abs().sum() > 0
    term.compute(.02)
    torch.testing.assert_close(env.joints, joints)
    torch.testing.assert_close(env.targets, targets)


def test_partial_reset_and_manual_selection_are_scoped():
    _, term = command_env()
    term.set_velocity(1., -1., 9., torch.tensor([1, 3]))
    assert term.requested_velocity[1].tolist() == pytest.approx([.25, -.25, math.pi / 2])
    term.compute(.02)
    unchanged = term.command[3].clone()
    term.reset(torch.tensor([1]))
    assert not term.manual_control[1] and term.manual_control[3]
    assert term.command[1].eq(0).all()
    term.compute(0., torch.tensor([1]))
    torch.testing.assert_close(term.command[3], unchanged)
    term.reset(slice(None))
    assert not term.manual_control.any() and term.command.eq(0).all()


@pytest.mark.parametrize("field,value", [
    ("smoothing_s", 0.), ("smoothing_s", float("inf")),
    ("frequency", -1.), ("frequency", float("nan")),
    ("velocity_limits", (.1, 0., .6)), ("velocity_limits", (.1, .1, float("inf"))),
    ("translation_probability", 1.1), ("translation_probability", .9),
    ("turn_probability", -.1), ("stop_probability", float("nan")),
])
def test_invalid_configuration_rejected(field, value):
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02)
    cfg = SnakeVelocityCommandCfg(resampling_time_range=(4., 8.))
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.build(cast(ManagerBasedRlEnv, env))


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_manual_velocity_rejected_without_mutation(value):
    _, term = command_env()
    with pytest.raises(ValueError):
        term.set_velocity(value, 0., 0.)
    assert term.requested_velocity.eq(0).all() and not term.manual_control.any()


def test_world_task_disables_sampling_without_changing_training_reset_flag():
    cfg = snake_env_cfg().commands['motion']
    assert cfg.training and not cfg.sample_velocity_commands
    env = SimpleNamespace(num_envs=8, device='cpu', step_dt=.02)
    term = cfg.build(cast(ManagerBasedRlEnv, env))
    term.reset(None)
    assert term.requested_velocity.eq(0).all()
    assert term.command.eq(0).all()
    assert term.time_left.isinf().all()


@pytest.mark.parametrize("play", [False, True])
def test_snake_velocity_visualization_enabled(play):
    assert snake_env_cfg(play=play).commands["motion"].debug_vis


@pytest.mark.parametrize("sign", [-1, 1])
def test_command_arrows_start_at_motion_site_without_height_offset(monkeypatch, sign):
    env, term = command_env(2)
    term.cfg.debug_vis = True
    env.cfg = SimpleNamespace(motion_site="foot_0_sole")
    positions = torch.tensor([[[1., 2., .05]], [[3., 4., .08]]])
    env.scene = {"robot": SimpleNamespace(
        data=SimpleNamespace(site_pos_w=positions),
        find_sites=lambda name: ([0], [name]),
    )}
    monkeypatch.setattr(mdp, "motion_axes", lambda env: (
        torch.tensor([[0., 1.], [0., 1.]]),
        torch.tensor([[-1., 0.], [-1., 0.]]),
    ))
    # 显示已平滑的有效指令，而不是尚未生效的手动目标。
    term._command[1] = torch.tensor([.125, .25, math.pi / 2]) * sign
    term.requested_velocity.zero_()
    arrows = []
    markers = []
    visualizer = SimpleNamespace(
        get_env_indices=lambda count: [1],
        add_arrow=lambda start, end, **kwargs: arrows.append((start, end)),
        add_sphere=lambda center, **kwargs: markers.append(center),
    )
    before = term.command.clone()
    term.debug_vis(visualizer)
    assert len(arrows) == 2
    assert len(markers) == 1
    origin = torch.tensor([3., 4., .08])
    for start, _ in arrows:
        torch.testing.assert_close(torch.as_tensor(start), origin)
    torch.testing.assert_close(torch.as_tensor(arrows[0][1]), origin + sign * torch.tensor([-.15, .075, 0.]))
    torch.testing.assert_close(torch.as_tensor(arrows[1][1]), origin + sign * torch.tensor([0., 0., .15]))
    torch.testing.assert_close(term.command, before)
    term._command.zero_()
    arrows.clear()
    term.debug_vis(visualizer)
    assert not arrows
    assert len(markers) == 2
    # 一个按键步长也能看到箭头，停车仍只显示基点。
    term._command[1, 0] = .025 * sign
    term.debug_vis(visualizer)
    assert len(arrows) == 1
    torch.testing.assert_close(torch.as_tensor(arrows[0][1]), origin + sign * torch.tensor([0., .04, 0.]))
    arrows.clear()
    markers.clear()
    term._command.fill_(.1)
    visualizer.get_env_indices = lambda count: []
    term.debug_vis(visualizer)
    assert not arrows
    assert not markers
