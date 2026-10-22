"""在 CPU 上对完整基座重力方向（包括基座 Z 轴）执行正向运动学回归检查。"""

import math
from types import SimpleNamespace

import mujoco
import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.experiments.transformation import (
    snake_to_biped_env_cfg as _recipe_snake_to_biped_env_cfg,
)
from zbot_rl_mjlab.robot import DEFAULT_JOINT_POS, JOINT_NAMES, get_spec
from zbot_rl_mjlab.tasks.transformation import snake_to_biped as reverse
from zbot_rl_mjlab.tasks.transformation.poses import SNAKE_ROOT_QUAT


def fk_environment(joints=DEFAULT_JOINT_POS, root=(1, 0, 0, 0)):
    model = get_spec().compile()
    native = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    native.qpos[3:7] = root
    for name, value in zip(JOINT_NAMES, joints, strict=True):
        native.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, native)  # pyright: ignore[reportAttributeAccessIssue]
    quaternion = torch.tensor(
        native.xquat[model.body("base").id].copy(), dtype=torch.float64
    )
    robot = SimpleNamespace(
        find_bodies=lambda name: ([0], [name]),
        data=SimpleNamespace(body_link_quat_w=quaternion.reshape(1, 1, 4)),
    )
    return SimpleNamespace(scene={"robot": robot}, cfg=_recipe_snake_to_biped_env_cfg())


def test_target_upright_has_zero_full_gravity_cost():
    env = fk_environment()
    assert reverse.base_gravity_alignment_cost(env).item() == pytest.approx(
        0, abs=1e-12
    )
    reference = reverse.standing_gravity_reference(tuple(DEFAULT_JOINT_POS))
    assert sum(component**2 for component in reference) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "joints,root",
    [
        ((0.0,) * 6, SNAKE_ROOT_QUAT),
        ((0.033083, -0.565864, 1.860875, -1.860895, 0.566303, -0.033464), (1, 0, 0, 0)),
        (DEFAULT_JOINT_POS, SNAKE_ROOT_QUAT),
    ],
)
def test_initial_wrong_middle_and_target_lying_sideways_have_positive_cost(
    joints, root
):
    env = fk_environment(joints, root)
    cost = reverse.base_gravity_alignment_cost(env).item()
    assert 0.1 < cost <= 2


def test_sideways_target_sees_full_gravity_difference_missed_by_xy_roll():
    env = fk_environment(DEFAULT_JOINT_POS, SNAKE_ROOT_QUAT)
    assert reverse.base_gravity_alignment_cost(env).item() == pytest.approx(
        1.0, abs=1e-9
    )
    assert reverse.base_z_roll_cost(env).item() < 0.2


def test_world_yaw_does_not_change_full_gravity_cost():
    env = fk_environment((0.0,) * 6, SNAKE_ROOT_QUAT)
    before = reverse.base_gravity_alignment_cost(env)
    yaw = torch.tensor([math.cos(0.7), 0.0, 0.0, math.sin(0.7)], dtype=torch.float64)
    data = env.scene["robot"].data
    data.body_link_quat_w[:] = quat_mul(yaw.reshape(1, 1, 4), data.body_link_quat_w)
    torch.testing.assert_close(reverse.base_gravity_alignment_cost(env), before)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0])
def test_invalid_quaternion_receives_worst_cost(value):
    env = fk_environment()
    env.scene["robot"].data.body_link_quat_w[:] = value
    assert reverse.base_gravity_alignment_cost(env).item() == 2.0


def test_reference_is_cached_by_configured_standing_joint_target():
    env = fk_environment((0.0,) * 6)
    env.cfg.standing_joint_pos = (0.0,) * 6
    assert reverse.base_gravity_alignment_cost(env).item() == pytest.approx(
        0, abs=1e-12
    )
    assert reverse.standing_gravity_reference(
        env.cfg.standing_joint_pos
    ) is reverse.standing_gravity_reference(env.cfg.standing_joint_pos)
