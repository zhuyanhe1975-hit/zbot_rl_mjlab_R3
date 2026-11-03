"""Geometry and checkpoint boundaries for the new fixed module clocking."""

import math
from itertools import pairwise

import mujoco
import numpy as np
import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.robot import get_spec
from zbot_rl_mjlab.rz270_snake_teacher import Rz270SnakeTeacherRunner
from zbot_rl_mjlab.site_snake_teacher import SiteSnakeTeacherRunner
from zbot_rl_mjlab.tasks.snake.rz270 import reset_rz270_snake, rz270_spec

TASK = 'Mjlab-Zbot-6dof-Snake-RZ270-Teacher'
MODULES = ('foot_0', 'a2', 'a3', 'base', 'a5', 'a6')


@pytest.mark.parametrize('factory,angle', [(get_spec, math.pi), (rz270_spec, 3 * math.pi / 2)])
def test_zero_joint_modules_stay_collinear_with_the_requested_relative_clocking(factory, angle):
    model = factory().compile()
    data = mujoco.MjData(model)
    data.qpos[3:7] = (1., 0., 0., 0.)
    data.qpos[7:] = 0.
    mujoco.mj_forward(model, data)
    rotation = np.array(((math.cos(angle), -math.sin(angle), 0.),
                         (math.sin(angle), math.cos(angle), 0.), (0., 0., 1.)))
    for previous, current in pairwise(MODULES):
        a, b = model.body(previous).id, model.body(current).id
        relative = data.xmat[a].reshape(3, 3).T @ data.xmat[b].reshape(3, 3)
        np.testing.assert_allclose(relative, rotation, atol=1e-8)
    positions = np.array([data.xpos[model.body(name).id] for name in MODULES])
    # CAD offsets retain sub-millimetre tolerances rather than bending the chain.
    assert np.ptp(positions[:, :2], axis=0).max() < .001
    assert np.all(np.diff(positions[:, 2]) > .1)


def test_task_has_its_own_model_and_training_identity():
    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play)
        assert cfg.scene.entities['robot'].spec_fn is rz270_spec
        assert set(cfg.scene.entities['robot'].init_state.joint_pos.values()) == {0.}
        assert cfg.commands['transform'].mode == 'snake'
        assert cfg.events['reset_transition'].func is reset_rz270_snake
    assert load_env_cfg('Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher').scene.entities['robot'].spec_fn is not rz270_spec
    assert load_rl_cfg(TASK).experiment_name == 'zbot_snake_rz270_teacher'
    assert load_runner_cls(TASK) is Rz270SnakeTeacherRunner


def checkpoint(runner):
    return {'infos': {'privileged_omni_teacher': runner._metadata_schema() | {
        'velocity_limits': (.2, .2, 2.), 'joint_target_limit': 2.,
        'snake_target_limit': 2., 'max_target_speed': 2 * math.pi,
        'heading_range': (-math.pi, math.pi), 'forward_speed_cap': .2,
    }}}


def test_old_and_new_assembly_checkpoints_are_not_interchangeable():
    old, new = checkpoint(SiteSnakeTeacherRunner), checkpoint(Rz270SnakeTeacherRunner)
    Rz270SnakeTeacherRunner._validate_checkpoint(new)
    with pytest.raises(ValueError):
        Rz270SnakeTeacherRunner._validate_checkpoint(old)
    with pytest.raises(ValueError):
        SiteSnakeTeacherRunner._validate_checkpoint(new)


def test_default_snake_pose_is_grounded_and_head_faces_positive_x():
    cfg = load_env_cfg(TASK)
    initial = cfg.scene.entities['robot'].init_state
    model = rz270_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = initial.pos
    data.qpos[3:7] = initial.rot
    data.qpos[7:] = 0.
    mujoco.mj_forward(model, data)
    site = model.site('foot_0_sole').id
    forward = data.site_xmat[site].reshape(3, 3) @ np.array((0., 0., -1.))
    np.testing.assert_allclose(forward, (1., 0., 0.), atol=1e-8)
    vertices = []
    for geom in range(model.ngeom):
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        count = model.mesh_vertnum[mesh]
        vertices.append(model.mesh_vert[start:start + count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    points = np.concatenate(vertices)
    assert .002 < points[:, 2].min() < .004
    assert points[:, 2].max() < .12
