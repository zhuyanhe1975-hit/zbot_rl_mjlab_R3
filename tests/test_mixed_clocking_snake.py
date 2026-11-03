"""Geometry, task inheritance, and checkpoint boundaries for mixed clocking."""
import math
from itertools import pairwise

import mujoco
import numpy as np
import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.mixed_clocking_snake_teacher import MixedClockingSnakeTeacherRunner
from zbot_rl_mjlab.rz270_snake_teacher import Rz270SnakeTeacherRunner
from zbot_rl_mjlab.tasks.snake.mixed_clocking import CLOCKING_DEG, mixed_clocking_spec
from zbot_rl_mjlab.tasks.snake.rz270 import reset_rz270_snake

TASK = 'Mjlab-Zbot-6dof-Snake-Mixed-270-270-180-90-90-Teacher'
OLD = 'Mjlab-Zbot-6dof-Snake-RZ270-Teacher'


def checkpoint(runner):
    return {'infos': {'privileged_omni_teacher': runner._metadata_schema() | {
        'velocity_limits': (.2, .2, .5), 'joint_target_limit': 2., 'snake_target_limit': 2.,
        'max_target_speed': 2 * math.pi, 'heading_range': (-math.pi, math.pi), 'forward_speed_cap': .2,
    }}}


def test_actual_five_local_clockings_and_grounded_reset():
    cfg = load_env_cfg(TASK)
    initial = cfg.scene.entities['robot'].init_state
    m = mixed_clocking_spec().compile(); d = mujoco.MjData(m)
    d.qpos[:3] = initial.pos; d.qpos[3:7] = initial.rot; d.qpos[7:] = 0.
    mujoco.mj_forward(m, d)
    for (left, right), angle in zip(pairwise(('foot_0','a2','a3','base','a5','a6')), CLOCKING_DEG, strict=True):
        relative = d.xmat[m.body(left).id].reshape(3,3).T @ d.xmat[m.body(right).id].reshape(3,3)
        a = math.radians(angle)
        np.testing.assert_allclose(relative, [[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]], atol=1e-8)
    site = m.site('foot_0_sole').id
    np.testing.assert_allclose(d.site_xmat[site].reshape(3,3) @ [0.,0.,-1.], [1.,0.,0.], atol=1e-8)
    assert not any(c.dist < -.0001 for c in d.contact)


def test_task_inherits_teacher_recipe_and_is_independent():
    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play); old = load_env_cfg(OLD, play=play)
        assert cfg.scene.entities['robot'].spec_fn is mixed_clocking_spec
        assert cfg.events['reset_transition'].func is reset_rz270_snake
        assert {k: v.weight for k,v in cfg.rewards.items()} == {k: v.weight for k,v in old.rewards.items()}
        assert cfg.commands['transform'].heading_range == old.commands['transform'].heading_range
    assert load_runner_cls(TASK) is MixedClockingSnakeTeacherRunner
    assert load_rl_cfg(TASK).experiment_name == 'zbot_snake_mixed_270_270_180_90_90_teacher'


def test_checkpoints_do_not_cross_assemblies():
    mixed, old = checkpoint(MixedClockingSnakeTeacherRunner), checkpoint(Rz270SnakeTeacherRunner)
    MixedClockingSnakeTeacherRunner._validate_checkpoint(mixed)
    with pytest.raises(ValueError): MixedClockingSnakeTeacherRunner._validate_checkpoint(old)
    with pytest.raises(ValueError): Rz270SnakeTeacherRunner._validate_checkpoint(mixed)
