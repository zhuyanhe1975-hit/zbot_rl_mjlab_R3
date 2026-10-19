"""低弧线使用目标分支，同时保留站立目标。"""

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.experiments.transformation import SnakeToBipedArc, SnakeToBipedBeta
from zbot_rl_mjlab.robot import DEFAULT_JOINT_POS, JOINT_NAMES, get_spec
from zbot_rl_mjlab.tasks.transformation.actions import bounded_target
from zbot_rl_mjlab.tasks.transformation.poses import (
    ARC_JOINT_POS,
    ARC_ROOT_POS,
    ARC_ROOT_QUAT,
    SNAKE_JOINT_POS,
)
from zbot_rl_mjlab.tasks.transformation.snake_to_biped import (
    SnakeToBipedEnvCfg,
    initial_joint_pose_cost,
)


@pytest.mark.parametrize("play", [False, True])
def test_arc_reset_and_branch_are_separate_from_the_original_beta(play):
    cfg = load_env_cfg(SnakeToBipedArc.task_id, play=play)
    original = load_env_cfg(SnakeToBipedBeta.task_id, play=play)
    pose = cfg.scene.entities["robot"].init_state
    assert pose.pos == ARC_ROOT_POS
    assert pose.rot == ARC_ROOT_QUAT
    assert pose.joint_pos == dict(zip(JOINT_NAMES, ARC_JOINT_POS, strict=True))
    assert cfg.initial_joint_pos == ARC_JOINT_POS
    assert original.initial_joint_pos == SNAKE_JOINT_POS
    assert original.scene.entities["robot"].init_state.joint_pos == dict(zip(JOINT_NAMES, SNAKE_JOINT_POS, strict=True))
    assert cfg.standing_joint_pos == original.standing_joint_pos == DEFAULT_JOINT_POS
    assert cfg.terminations == original.terminations
    assert cfg.sim == original.sim
    assert cfg.observations == original.observations
    assert cfg.scene.entities["robot"].articulation == original.scene.entities["robot"].articulation
    bounds = cfg.actions["joint_pos"].joint_target_bounds
    assert len(bounds) == 6
    for name, initial, target in zip(JOINT_NAMES, ARC_JOINT_POS, DEFAULT_JOINT_POS, strict=True):
        low, high = bounds[name]
        assert low <= initial <= high and low <= target <= high
        assert -math.pi <= low <= high <= math.pi
        assert low == 0 if target > 0 else high == 0
    assert len(original.actions["joint_pos"].joint_target_bounds) == 2
    assert load_rl_cfg(SnakeToBipedArc.task_id).actor.distribution_cfg["class_name"] == "BetaDistribution"


@pytest.mark.parametrize("direction", [-1, 1])
def test_repeated_extreme_commands_stay_on_target_branch(direction):
    cfg = SnakeToBipedArc().build_env()
    assert isinstance(cfg, SnakeToBipedEnvCfg)
    limits = torch.tensor([cfg.actions["joint_pos"].joint_target_bounds[name] for name in JOINT_NAMES])
    command = torch.tensor(ARC_JOINT_POS)
    action = torch.full((6,), direction, dtype=torch.float32)
    for _ in range(400):
        command = bounded_target(command, action, limits, 0.02)
        assert torch.all(command * torch.tensor(DEFAULT_JOINT_POS) >= 0)
        assert command.abs().le(math.pi).all()
    torch.testing.assert_close(command, limits[:, 0 if direction < 0 else 1])


def test_initial_pose_penalty_uses_the_selected_arc_instead_of_zero_angles():
    positions = torch.tensor([ARC_JOINT_POS, SNAKE_JOINT_POS])
    env = SimpleNamespace(
        cfg=SnakeToBipedArc().build_env(),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=positions))},
    )
    cost = initial_joint_pose_cost(env)
    torch.testing.assert_close(cost, torch.tensor([0, sum(q**2 for q in ARC_JOINT_POS) / 6]))


def test_shallow_arc_mass_projection_stays_within_one_cm_of_the_foot_segment():
    model = get_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = ARC_ROOT_POS
    data.qpos[3:7] = ARC_ROOT_QUAT
    data.qpos[7:] = ARC_JOINT_POS
    mujoco.mj_forward(model, data)
    com = np.sum(data.xipos * model.body_mass[:, None], axis=0) / model.body_mass.sum()
    sites = data.site_xpos[[model.site(n).id for n in ("foot_0_sole", "foot_1_sole")], :2]
    segment = sites[1] - sites[0]
    fraction = (com[:2] - sites[0]) @ segment / (segment @ segment)
    assert 0 < fraction < 1
    assert np.linalg.norm(com[:2] - sites[0] - fraction * segment) < 0.01
    assert com[2] < 0.06
