"""The base target moves independently of the head while retaining command axes."""
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul
from test_site_snake_heading import fixture

from zbot_rl_mjlab.robot import get_spec
from zbot_rl_mjlab.tasks.snake import site_omni as task
from zbot_rl_mjlab.tasks.snake.base_omni import (
    base_motion_spec,
)
from zbot_rl_mjlab.tasks.universal import mdp, omni

mj: Any = mujoco  # MuJoCo exposes these bindings dynamically.


@pytest.mark.parametrize('yaw', [0., np.pi/2, np.pi, -np.pi/2])
def test_asset_reference_site_is_at_base_with_initial_forward_and_up_preserved(yaw):
    original = get_spec().compile()
    assert mj.mj_name2id(original, mj.mjtObj.mjOBJ_SITE, 'base_motion') == -1
    model = base_motion_spec().compile()
    assert model.nsite == original.nsite+1
    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    np.testing.assert_array_equal(model.body_inertia, original.body_inertia)
    np.testing.assert_array_equal(model.geom_friction, original.geom_friction)
    data = mj.MjData(model)
    data.qpos[3:7] = quat_mul(torch.tensor([np.cos(yaw/2), 0., 0., np.sin(yaw/2)], dtype=torch.float64),
                              torch.tensor([2**-.5, 0., -2**-.5, 0.], dtype=torch.float64)).numpy()
    mj.mj_forward(model, data)
    base = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'base')
    site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'base_motion')
    head = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'foot_0_sole')
    assert model.site_bodyid[site] == base
    np.testing.assert_allclose(data.site_xpos[site], data.xpos[base], atol=1e-8)
    np.testing.assert_allclose(data.site_xmat[site], data.site_xmat[head], atol=1e-6)
    np.testing.assert_allclose(data.site_xmat[site].reshape(3, 3)[:, 0], [0., 0., 1.], atol=1e-6)


def two_site_env(yaw=0.) -> tuple[Any, Any]:
    env, term = fixture((yaw,))
    data = env.scene['robot'].data
    for name in ('site_quat_w', 'site_lin_vel_w', 'site_ang_vel_w'):
        setattr(data, name, getattr(data, name).repeat(1, 2, 1))
    # The old head is index0; moving the base site to index1 must affect all paths.
    env.scene['robot'].find_sites = lambda name: ([1 if name=='base_motion' else 0], [name])
    env.cfg.motion_site = 'base_motion'
    del env._omni_motion_site_index
    return env, term


@pytest.mark.parametrize('yaw', [0., .7, -2.1])
def test_velocity_reward_and_frame_ignore_head_translation_and_rotation(yaw):
    env, term = two_site_env(yaw)
    data = env.scene['robot'].data
    heading = omni.site_heading(env).clone()
    term.velocity_command[:, 0] = .1
    data.site_lin_vel_w[:, 1, :2] = .1*heading
    data.site_ang_vel_w[:, 1, 2] = .2
    data.site_lin_vel_w[:, 0] = torch.tensor([99., 99., 99.])
    data.site_ang_vel_w[:, 0] = torch.tensor([99., 99., 99.])
    data.site_quat_w[:, 0] = torch.tensor([1., 0., 0., 0.])
    torch.testing.assert_close(mdp.tracking_velocity(env), torch.tensor([[.1, 0., .2]]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(task.forward_tracking(env), torch.ones(1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(omni.site_heading(env), heading)
    data.site_lin_vel_w[:, 1] = 0.
    expected = -1.
    assert task.forward_tracking(env).item() == pytest.approx(expected)


def test_base_yaw_rotates_reference_while_head_yaw_stays_fixed():
    env, _ = two_site_env()
    data = env.scene['robot'].data
    old_head = data.site_quat_w[:, 0].clone()
    yaw = .8
    q = torch.tensor([[np.cos(yaw/2), 0., 0., np.sin(yaw/2)]], dtype=torch.float32)
    data.site_quat_w[:, 1] = quat_mul(q, data.site_quat_w[:, 1])
    heading = omni.site_heading(env)
    torch.testing.assert_close(heading, torch.tensor([[np.cos(yaw), np.sin(yaw)]], dtype=torch.float32))
    torch.testing.assert_close(data.site_quat_w[:, 0], old_head)


def test_base_gravity_and_height_ignore_a_flipped_high_head():
    class Scene(dict):
        env_origins: torch.Tensor

    env, _ = two_site_env()
    data = env.scene['robot'].data
    data.site_quat_w[:, 0] = torch.tensor([2**-.5, 0., 2**-.5, 0.])
    data.site_pos_w = torch.tensor([[[0., 0., 10.], [0., 0., .06]]])
    env.scene = Scene(env.scene)
    env.scene.env_origins = torch.zeros(1, 3)
    torch.testing.assert_close(task.site_projected_gravity(env), torch.tensor([[-1., 0., 0.]]), atol=1e-6, rtol=0)
    assert not task.site_tilt_limit_reached(env).any()
    assert task.site_height(env).item() == pytest.approx(.06)


def test_old_universal_default_still_resolves_head_site():
    selected = []
    env = SimpleNamespace(cfg=SimpleNamespace(), scene={'robot': SimpleNamespace(
        find_sites=lambda name: (selected.append(name) or [3], [name]))})
    assert omni.motion_site_index(env) == 3
    assert selected == ['foot_0_sole']


def test_experiment20_is_isolated_and_retains_experiment19_logic():
    from dataclasses import asdict

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    from zbot_rl_mjlab.adaptive_site_snake import AdaptiveSiteSnakeTeacherRunner
    from zbot_rl_mjlab.base_snake_teacher import BaseSnakeAdaptiveTeacherRunner

    old = cast(task.SiteSnakeEnvCfg, load_env_cfg('Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher', play=True))
    new = cast(task.SiteSnakeEnvCfg, load_env_cfg('Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher', play=True))
    assert old.motion_site == 'foot_0_sole' and new.motion_site == 'base_motion'
    velocity_names = ('track_forward_velocity', 'track_lateral_velocity', 'track_angular_velocity')
    assert old.commands == new.commands and old.rewards.keys() == new.rewards.keys()
    for name in old.rewards:
        if name in velocity_names:
            assert new.rewards[name].func is old.rewards[name].func
            assert new.rewards[name].weight == old.rewards[name].weight
        else:
            assert old.rewards[name] == new.rewards[name]
    assert asdict(old.actions['joint_pos']) == asdict(new.actions['joint_pos'])
    assert old.sim == new.sim and old.events == new.events
    assert old.observations == new.observations and old.terminations == new.terminations
    assert old.scene.entities['robot'].spec_fn != new.scene.entities['robot'].spec_fn
    assert load_runner_cls('Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher') is AdaptiveSiteSnakeTeacherRunner
    assert load_runner_cls('Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher') is BaseSnakeAdaptiveTeacherRunner
    assert load_rl_cfg('Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher').experiment_name != load_rl_cfg('Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher').experiment_name


def test_new_checkpoint_role_keeps_old_head_contract_intact():
    from zbot_rl_mjlab.adaptive_site_snake import AdaptiveSiteSnakeTeacherRunner
    from zbot_rl_mjlab.base_snake_teacher import BaseSnakeAdaptiveTeacherRunner

    controls = {'velocity_limits': (.5, .5, .25), 'joint_target_limit': torch.pi/2,
                'snake_target_limit': torch.pi/2, 'max_target_speed': 2*torch.pi,
                'velocity_std_fractions': (.25, .25, .25)}
    head = {'infos': {'privileged_omni_teacher': AdaptiveSiteSnakeTeacherRunner._metadata_schema() | controls}}
    base = {'infos': {'privileged_omni_teacher': BaseSnakeAdaptiveTeacherRunner._metadata_schema() | controls}}
    assert AdaptiveSiteSnakeTeacherRunner._validate_checkpoint(head)['tracking_point'] == 'foot_0_sole'
    assert BaseSnakeAdaptiveTeacherRunner._validate_checkpoint(base)['tracking_point'] == 'base'
    for runner, saved in ((AdaptiveSiteSnakeTeacherRunner, base), (BaseSnakeAdaptiveTeacherRunner, head)):
        with pytest.raises(ValueError, match='compatible'):
            runner._validate_checkpoint(saved)


@pytest.mark.parametrize('play', [False, True])
def test_experiment20_visual_offset_and_measured_arrows_are_play_only(play):
    from mjlab.tasks.registry import load_env_cfg

    old = cast(task.SiteSnakeEnvCfg, load_env_cfg('Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher', play=play))
    new = cast(task.SiteSnakeEnvCfg, load_env_cfg('Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher', play=play))
    assert old.command_visual_height == 0. and not old.command_visual_actual_velocity
    assert new.command_visual_height == (.1 if play else 0.)
    assert new.command_visual_actual_velocity == play


def test_play_draws_command_and_measured_velocity_above_base_without_moving_tracking_point():
    from zbot_rl_mjlab.experiments.site_snake import base_snake_env_cfg

    env, term = two_site_env()
    env.cfg = base_snake_env_cfg(play=True)
    data = env.scene['robot'].data
    data.site_pos_w = torch.tensor([[[9., 9., 9.], [1., 2., .06]]])
    original = data.site_pos_w.clone()
    data.site_lin_vel_w[:, 1] = torch.tensor([-.05, .02, 0.])
    data.site_ang_vel_w[:, 1, 2] = -.1
    term.velocity_command[:] = torch.tensor([.1, 0., .2])
    markers, arrows = [], []
    visualizer = SimpleNamespace(env_idx=0, get_env_indices=lambda _: [0],
        add_sphere=lambda center, **kwargs: markers.append(np.array(center).copy()),
        add_arrow=lambda start, end, **kwargs: arrows.append((np.array(start).copy(), np.array(end).copy(), kwargs)))
    term._debug_vis_impl(visualizer)
    assert len(markers) == 1 and len(arrows) == 4
    np.testing.assert_allclose(markers[0], [1., 2., .16], atol=1e-6)
    for start, _, _ in arrows:
        np.testing.assert_allclose(start, markers[0])
    assert arrows[0][1][0] > arrows[0][0][0]  # Forward command, backward measured motion.
    assert arrows[2][1][0] < arrows[2][0][0]
    assert arrows[1][1][2] > arrows[1][0][2]  # Left-turn command, right-turn actual.
    assert arrows[3][1][2] < arrows[3][0][2]
    assert arrows[0][2]['color'] != arrows[2][2]['color']
    assert arrows[1][2]['color'] != arrows[3][2]['color']
    torch.testing.assert_close(data.site_pos_w, original)
    torch.testing.assert_close(mdp.tracking_velocity(env), torch.tensor([[-.05, .02, -.1]]), atol=1e-6, rtol=1e-5)
