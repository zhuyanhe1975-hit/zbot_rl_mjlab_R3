"""仅使用 CPU 检查任务内部的指令目标，不依赖物理限位。"""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction

from zbot_rl_mjlab.tasks.transformation.actions import (
    JointTargetRateAction,
    JointTargetRateActionCfg,
)


@pytest.fixture
def make_action(monkeypatch):
    def initialize(self, cfg, env):
        self._env = env
        self._target_names = [f"joint{i}" for i in range(1, 7)]
        self._target_ids = list(range(6))
        self._raw_actions = torch.zeros(2, 6)
        self._processed_actions = torch.zeros(2, 6)
        self._entity = SimpleNamespace(data=env.data)

    monkeypatch.setattr(JointPositionAction, "__init__", initialize)

    def make(bounds=None, *, target_limit=None, limits=None, initial=0.0):
        data = SimpleNamespace(
            default_joint_pos=torch.full((2, 6), initial),
            joint_pos=torch.zeros(2, 6),
            joint_pos_limits=(
                torch.tensor([-6.28, 6.28]).expand(2, 6, 2).clone()
                if limits is None
                else limits
            ),
        )
        cfg = JointTargetRateActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            joint_target_bounds=bounds or {},
            target_limit=target_limit,
            max_target_speed=1000.0,
        )
        return JointTargetRateAction(cfg, SimpleNamespace(data=data, step_dt=0.02))

    return make


def test_only_named_ends_are_bounded_and_physical_limits_are_unchanged(make_action):
    action = make_action({"joint1": (-0.688, 1.312), "joint6": (-1.312, 0.688)})
    original = action._entity.data.joint_pos_limits.clone()
    action.process_actions(torch.ones(2, 6))
    torch.testing.assert_close(
        action.commanded_target,
        torch.tensor([[1.312, 6.28, 6.28, 6.28, 6.28, 0.688]]).expand(2, 6),
    )
    action.process_actions(-torch.ones(2, 6))
    torch.testing.assert_close(
        action.commanded_target,
        torch.tensor([[-0.688, -6.28, -6.28, -6.28, -6.28, -1.312]]).expand(2, 6),
    )
    torch.testing.assert_close(action._entity.data.joint_pos_limits, original)


def test_initial_and_selected_reset_targets_are_clamped(make_action):
    action = make_action({"joint1": (-0.5, 0.5)}, initial=2.0)
    assert action.commanded_target[:, 0].eq(0.5).all()
    assert action.commanded_target[:, 1:].eq(2.0).all()
    action._entity.data.joint_pos[1] = -2.0
    action.reset(torch.tensor([1]))
    assert action.commanded_target[0, 0] == 0.5
    assert action.commanded_target[1, 0] == -0.5
    assert action.commanded_target[1, 1:].eq(-2.0).all()


def test_intersects_named_physical_and_global_target_limits(make_action):
    limits = torch.tensor([-3.0, 3.0]).expand(2, 6, 2).clone()
    limits[:, 0] = torch.tensor([-0.2, 0.4])
    action = make_action({"joint1": (-0.1, 0.8)}, target_limit=0.3, limits=limits)
    action.process_actions(torch.ones(2, 6))
    assert action.commanded_target.eq(0.3).all()
    action.process_actions(-torch.ones(2, 6))
    assert action.commanded_target[:, 0].eq(-0.1).all()
    assert action.commanded_target[:, 1:].eq(-0.3).all()


def test_long_saturation_stores_clipped_state_and_reverses_immediately(make_action):
    action = make_action(target_limit=math.pi / 2)
    action.rate_cfg.max_target_speed = 2 * math.pi
    direction = torch.tensor([[1., -1., 1., -1., 1., -1.],
                              [-1., 1., -1., 1., -1., 1.]])
    persistent_target = action.commanded_target
    pointer = persistent_target.data_ptr()
    for _ in range(1000):
        action.process_actions(direction)
    torch.testing.assert_close(persistent_target, direction * (math.pi / 2))
    action.process_actions(-direction)
    expected = direction * (math.pi / 2 - 2 * math.pi * .02)
    torch.testing.assert_close(action.commanded_target, expected)
    assert action.commanded_target.data_ptr() == pointer
    assert torch.rad2deg(action.commanded_target[0, 0]).item() == pytest.approx(82.8)
    action.process_actions(torch.zeros_like(direction))
    torch.testing.assert_close(persistent_target, expected)


def test_global_target_limit_also_clamps_initial_and_selected_reset_state(make_action):
    action = make_action(target_limit=.5, initial=2.)
    assert action.commanded_target.eq(.5).all()
    action._entity.data.joint_pos[1] = -2.
    action.reset(torch.tensor([1]))
    assert action.commanded_target[0].eq(.5).all()
    assert action.commanded_target[1].eq(-.5).all()
    assert action._entity.data.joint_pos[1].eq(-2.).all()


@pytest.mark.parametrize(
    "bounds",
    [
        {"unknown": (0.0, 1.0)},
        {"joint1": (1.0, -1.0)},
        {"joint1": (float("nan"), 1.0)},
        {"joint1": (0.0, float("inf"))},
        {"joint1": (7.0, 8.0)},
    ],
)
def test_invalid_bounds_fail_at_construction(make_action, bounds):
    with pytest.raises(ValueError):
        make_action(bounds)


def test_disjoint_global_limit_fails_at_construction(make_action):
    with pytest.raises(ValueError, match="do not intersect"):
        make_action({"joint1": (2.0, 3.0)}, target_limit=1.0)


def test_empty_bounds_preserve_initial_and_reset_behavior(make_action):
    action = make_action(initial=7.0)
    assert action.commanded_target.eq(7.0).all()
    action._entity.data.joint_pos[:] = -7.0
    action.reset()
    assert action.commanded_target.eq(-7.0).all()
