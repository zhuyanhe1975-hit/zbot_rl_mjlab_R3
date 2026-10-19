"""初始横滚轴、碰撞间隙与选择性重置的测试覆盖。"""

import math
from types import SimpleNamespace

import mujoco
import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply

from zbot_rl_mjlab.experiments.transformation import (
    SnakeToBipedRobust,
    SnakeToBipedTeacher,
)
from zbot_rl_mjlab.robot import get_spec
from zbot_rl_mjlab.tasks.transformation.poses import SNAKE_ROOT_POS, SNAKE_ROOT_QUAT
from zbot_rl_mjlab.tasks.transformation.roll_reset import (
    collision_radial_envelope,
    randomize_initial_snake_roll,
    rolled_root_states,
)


def default_states(count, dtype=torch.float64):
    states = torch.zeros(count, 13, dtype=dtype)
    states[:, :3] = torch.tensor(SNAKE_ROOT_POS, dtype=dtype)
    states[:, 3:7] = torch.tensor(SNAKE_ROOT_QUAT, dtype=dtype)
    return states


@pytest.mark.parametrize("half_range_deg", [90., 180.])
def test_roll_is_about_horizontal_snake_axis_and_has_mesh_plane_clearance(half_range_deg):
    half_range = math.radians(half_range_deg)
    angles = torch.linspace(-half_range, half_range, 73, dtype=torch.float64)
    radial, low_z, high_z = collision_radial_envelope((0,) * 6)
    origins = torch.zeros(73, 3, dtype=torch.float64)
    origins[:, 2] = torch.linspace(0, 0.2, 73)
    states = rolled_root_states(default_states(73), origins, angles, torch.tensor(radial, dtype=torch.float64), low_z, high_z, 0.0001)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(73, 3)
    torch.testing.assert_close(quat_apply(states[:, 3:7], axis), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64).expand(73, 3))
    assert states[:, 7:].eq(0).all()
    model = get_spec().compile()
    data = mujoco.MjData(model)
    for state, origin in zip(states, origins, strict=True):
        data.qpos[:7] = state[:7].numpy()
        data.qpos[7:] = 0
        mujoco.mj_forward(model, data)
        lowest = math.inf
        for geom in range(model.ngeom):
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[start : start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            lowest = min(lowest, float(points[:, 2].min()))
        assert lowest - origin[2].item() == pytest.approx(0.0001, abs=1e-7)


@pytest.fixture
def roll_env():
    class Scene(dict):
        pass

    defaults = default_states(32)
    written = defaults.clone()

    def write(states, env_ids):
        written[env_ids] = states

    robot = SimpleNamespace(data=SimpleNamespace(default_root_state=defaults, default_joint_pos=torch.zeros(32, 6)), write_root_state_to_sim=write)
    scene = Scene(robot=robot)
    scene.env_origins = torch.arange(32, dtype=defaults.dtype)[:, None] * torch.tensor([[4, 0, 0]])
    return SimpleNamespace(num_envs=32, device="cpu", cfg=SnakeToBipedRobust().build_env(), scene=scene, written=written)


def test_reset_is_random_reproducible_and_only_changes_selected_worlds(roll_env):
    env = roll_env
    defaults = env.scene["robot"].data.default_root_state.clone()
    torch.manual_seed(42)
    randomize_initial_snake_roll(env, None)
    offsets = env._initial_snake_roll_offset.clone()
    assert offsets.min() < -1 and offsets.max() > 1
    assert offsets.abs().max() <= math.pi / 2
    previous = env.written.clone()
    ids = torch.tensor([2, 7])
    randomize_initial_snake_roll(env, ids)
    others = torch.ones(32, dtype=torch.bool)
    others[ids] = False
    torch.testing.assert_close(env.written[others], previous[others])
    assert not torch.equal(env.written[ids, 3:7], previous[ids, 3:7])
    torch.testing.assert_close(env.scene["robot"].data.default_root_state, defaults)
    torch.manual_seed(42)
    randomize_initial_snake_roll(env, None)
    torch.testing.assert_close(env._initial_snake_roll_offset, offsets)
    env.cfg.initial_roll_range_deg = (0, 0)
    randomize_initial_snake_roll(env, None)
    torch.testing.assert_close(env.written[:, 3:7], defaults[:, 3:7])


@pytest.mark.parametrize("bounds", [(1, -1), (-181, 0), (0, float("nan"))])
def test_invalid_roll_range_is_rejected(roll_env, bounds):
    roll_env.cfg.initial_roll_range_deg = bounds
    with pytest.raises(ValueError):
        randomize_initial_snake_roll(roll_env, None)


@pytest.mark.parametrize("experiment", [SnakeToBipedRobust, SnakeToBipedTeacher])
def test_play_randomizes_seed_across_launches_and_training_keeps_reproducibility(experiment):
    training = experiment().build_env()
    playback = experiment().build_env(play=True)
    assert training.seed == 42
    assert playback.seed == -1
    assert playback.initial_roll_range_deg == (-90., 90.)
    assert playback.events["reset_transition"].func is training.events["reset_transition"].func
