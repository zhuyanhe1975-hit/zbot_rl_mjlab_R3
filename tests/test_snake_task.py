"""蛇形教师的坐标、相机稳定、推进投机与瞬态接触边界。"""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.experiments.snake import SnakeOmniTeacher
from zbot_rl_mjlab.robot import get_spec
from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg
from zbot_rl_mjlab.tasks.transformation.env_cfg import full_body_spec


class Scene(dict):
    env_origins = torch.zeros(2, 3)


def state_env():
    n, bodies = 2, 12
    cfg = snake_env_cfg(play=True)
    data = SimpleNamespace(
        joint_pos=torch.zeros(n, 6), default_joint_pos=torch.zeros(n, 6), joint_vel=torch.zeros(n, 6),
        actuator_force=torch.zeros(n, 6), root_link_pos_w=torch.zeros(n, 3),
        body_link_pos_w=torch.zeros(n, bodies, 3),
        body_com_pos_w=torch.zeros(n, bodies, 3),
        body_link_quat_w=torch.tensor(cfg.scene.entities['robot'].init_state.rot).repeat(n, bodies, 1),
        body_link_lin_vel_w=torch.zeros(n, bodies, 3),
        body_com_lin_vel_w=torch.zeros(n, bodies, 3),
        body_link_ang_vel_w=torch.zeros(n, bodies, 3),
    )
    data.body_link_pos_w[..., 2] = .055
    data.body_com_pos_w[..., 2] = .055
    # 此site在foot_0上无局部旋转，视图随刚体测试姿态同步更新。
    data.site_pos_w = data.body_link_pos_w[:, :1]
    data.site_quat_w = data.body_link_quat_w[:, :1]
    data.site_ang_vel_w = data.body_link_ang_vel_w[:, :1]
    robot = SimpleNamespace(data=data, find_bodies=lambda name: ([0], [name]),
                            find_sites=lambda name: ([0], [name]),
                            indexing=SimpleNamespace(body_ids=torch.arange(bodies)))
    ground = SimpleNamespace(found=torch.ones(n, bodies), force=torch.zeros(n, bodies, 3), force_history=torch.zeros(n, 4, bodies, 3))
    own = SimpleNamespace(found=torch.zeros(n, 1), force_history=torch.zeros(n, 4, 1, 3), dist_history=torch.zeros(n, 4, 1))
    scene = Scene(robot=robot, snake_ground_contact=SimpleNamespace(data=ground), snake_self_contact=SimpleNamespace(data=own))
    scene['robot/head_lin_acc'] = SimpleNamespace(data=torch.zeros(n, 3))
    scene['robot/head_ang_acc'] = SimpleNamespace(data=torch.zeros(n, 3))
    term = SimpleNamespace(command=torch.zeros(n, 3), requested_velocity=torch.zeros(n, 3), cfg=SimpleNamespace(frequency=.7, velocity_limits=(.25, .25, math.pi / 2)))
    action = SimpleNamespace(commanded_target=torch.zeros(n, 6), raw_action=torch.zeros(n, 6))
    return SimpleNamespace(
        num_envs=n, device='cpu', cfg=cfg, scene=scene, step_dt=.02, episode_length_buf=torch.zeros(n),
        command_manager=SimpleNamespace(get_term=lambda name: term),
        termination_manager=SimpleNamespace(get_term=lambda name: torch.zeros(n, dtype=torch.bool)),
        action_manager=SimpleNamespace(get_term=lambda name: action),
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(n, bodies))),
    )


def test_independent_teacher_uses_full_body_friction_without_changing_old_robot():
    cfg = snake_env_cfg()
    assert cfg.scene.entities['robot'].spec_fn is full_body_spec
    assert all(g.condim == 3 for g in full_body_spec().geoms if g.contype or g.conaffinity)
    assert all(g.condim == 1 for g in get_spec().geoms if g.contype or g.conaffinity)
    assert not cfg.curriculum
    assert set(cfg.commands) == {'motion'}
    assert 'reset_transition' not in cfg.events
    assert cfg.scene.num_envs == 4096
    assert snake_env_cfg(play=True).scene.num_envs == 1
    assert SnakeOmniTeacher().build_ppo().experiment_name == 'zbot_snake_omni_teacher'
    assert cfg.terminations['self_collision'].func is mdp.self_collision
    sensors = {s.name: s for s in cfg.scene.sensors}
    own = sensors['snake_self_contact']
    assert own.primary == own.secondary
    assert own.secondary_policy != 'any'
    assert own.history_length == cfg.decimation


def test_snake_only_limits_action_targets_and_preserves_model_ranges():
    from zbot_rl_mjlab.robot import JOINT_NAMES

    for play in (False, True):
        cfg = snake_env_cfg(play=play)
        model = cfg.scene.entities['robot'].spec_fn().compile()
        assert cfg.actions['joint_pos'].target_limit == pytest.approx(math.pi / 2)
        for name in JOINT_NAMES:
            joint = model.joint(name)
            assert model.jnt_limited[joint.id]
            assert model.jnt_range[joint.id].tolist() == pytest.approx([-6.28, 6.28])
    # 独立场景函数不得原位收窄变形任务的关节范围。
    original = full_body_spec().compile()
    assert original.joint('joint3').range.tolist() == pytest.approx([-6.28, 6.28])


def test_initial_xml_pillar_rotates_about_foot_0_y_and_stays_level():
    import mujoco
    import numpy as np

    cfg = snake_env_cfg(play=True)
    robot = cfg.scene.entities['robot']
    model = robot.spec_fn().compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = robot.init_state.pos
    data.qpos[3:7] = robot.init_state.rot
    data.qpos[7:] = 0
    mujoco.mj_forward(model, data)
    body_matrix = data.xmat[model.body('foot_0').id].reshape(3, 3)
    site_matrix = data.site_xmat[model.site(cfg.motion_site).id].reshape(3, 3)
    for matrix in (body_matrix, site_matrix):
        np.testing.assert_allclose(matrix[:, 2], [-1., 0., 0.], atol=1e-7)
        np.testing.assert_allclose(matrix[:, 0], [0., 0., 1.], atol=1e-7)
    np.testing.assert_allclose(body_matrix @ cfg.head_up_axis, [0., 0., 1.], atol=1e-7)
    env = state_env()
    assert mdp.head_tilt(env).max() < .001
    assert mdp.head_tilt_cost(env).max() < 1e-6
    assert not mdp.head_tilt_limit_reached(env).any()


def test_head_motion_forward_is_world_x_and_left_is_world_y():
    env = state_env()
    torch.testing.assert_close(mdp.head_forward(env), torch.tensor([[1., 0.], [1., 0.]]))
    env.scene['robot'].data.body_com_lin_vel_w[..., 0] = .05
    assert mdp.planar_velocity(env)[:, 0].tolist() == pytest.approx([.05, .05])
    env.scene['robot'].data.body_com_lin_vel_w[..., 0] = 0
    env.scene['robot'].data.body_com_lin_vel_w[..., 1] = .05
    assert mdp.planar_velocity(env)[:, 1].tolist() == pytest.approx([.05, .05])


def test_head_only_translation_does_not_count_as_whole_robot_speed():
    env = state_env()
    env.scene['robot'].data.body_link_lin_vel_w[:, 0, 0] = .2
    assert mdp.planar_velocity(env).eq(0).all()
    env.scene['robot'].data.body_com_lin_vel_w[:, 0, 0] = .12
    assert mdp.planar_velocity(env)[:, 0].tolist() == pytest.approx([.01, .01])


def test_head_stability_allows_commanded_vertical_axis_rotation():
    env = state_env()
    yaw = torch.tensor([[math.cos(.4), 0., 0., math.sin(.4)]]).expand(2, -1)
    q = quat_mul(yaw, env.scene['robot'].data.body_link_quat_w[:, 0])
    env.scene['robot'].data.body_link_quat_w[:, 0] = q
    assert mdp.head_tilt(env).max() < .001
    env.command_manager.get_term('motion').command[:, 2] = .3
    env.scene['robot'].data.body_link_ang_vel_w[:, 0, 2] = .3
    assert mdp.head_rotation_cost(env).eq(0).all()
    env.scene['robot'].data.body_link_ang_vel_w[:, 0, 0] = .3
    assert (mdp.head_rotation_cost(env) > 0).all()


@pytest.mark.parametrize('axis,sign', [(0,1), (0,-1), (1,1), (1,-1), (2,1), (2,-1)])
def test_tracking_rewards_net_progress_and_rejects_zero_net_oscillation(axis, sign):
    cmd = torch.zeros(1, 3)
    cmd[:, axis] = sign * (.05 if axis < 2 else .3)
    zero = torch.zeros_like(cmd)
    assert mdp.tracking_score(zero, cmd).item() == 0
    assert mdp.tracking_score(cmd, cmd).item() == pytest.approx(1.)
    assert mdp.tracking_score(-cmd, cmd).item() < 0
    average = (mdp.tracking_score(cmd, cmd) + mdp.tracking_score(-cmd, cmd)) / 2
    assert average.item() < 0


def test_contact_penalty_allows_static_support_but_detects_substep_impact():
    env = state_env()
    history = env.scene['snake_ground_contact'].data.force_history
    history[:, 0, 0, 2] = 12 * 9.81
    assert mdp.impact_cost(env).eq(0).all()
    history[:, 1, 0, 2] = 3 * 12 * 9.81
    assert mdp.impact_cost(env).min() > 0
    assert mdp.normalized_ground_peak(env).tolist() == pytest.approx([3., 3.])


def test_self_collision_catches_transient_force_and_penetration():
    env = state_env()
    own = env.scene['snake_self_contact'].data
    assert not mdp.self_collision(env).any()
    own.force_history[0, 1, 0, 0] = .1
    own.dist_history[1, 0, 0] = -.001
    assert mdp.self_collision(env).all()


def test_teacher_observation_and_safety_are_finite_at_standard_snake_pose():
    env = state_env()
    assert mdp.actor_state(env).shape == (2, 40)
    assert mdp.critic_state(env).shape == (2, 55)
    assert torch.isfinite(mdp.actor_state(env)).all()
    assert not mdp.unsafe(env).any()
    assert mdp.head_acceleration_cost(env).eq(0).all()
    env.scene['robot'].data.root_link_pos_w[0, 0] = env.cfg.workspace_radius + 1
    env.scene['robot'].data.joint_pos[1, 0] = float('nan')
    assert mdp.unsafe(env).all()


def test_joint_pose_penalty_uses_initial_angles_and_does_not_cancel_opposite_bends():
    env = state_env()
    data = env.scene['robot'].data
    assert mdp.joint_pose_deviation_cost(env).eq(0).all()
    data.default_joint_pos[:] = .1
    data.joint_pos.copy_(data.default_joint_pos)
    assert mdp.joint_pose_deviation_cost(env).eq(0).all()
    offset = torch.tensor([.3, -.3, .3, -.3, .3, -.3])
    data.joint_pos[:] = data.default_joint_pos + offset
    assert mdp.joint_pose_deviation_cost(env).tolist() == pytest.approx([.09, .09])
    assert mdp.joint_pose_rms(env).tolist() == pytest.approx([.3, .3])
    assert mdp.joint_pose_max_deviation(env).tolist() == pytest.approx([.3, .3])
    data.joint_pos[:] = data.default_joint_pos + 4 * offset
    assert mdp.joint_pose_deviation_cost(env).tolist() == pytest.approx([1.44, 1.44])


def test_joint_pose_penalty_and_diagnostic_are_independent_of_world_heading():
    env = state_env()
    cfg = env.cfg
    assert cfg.rewards['joint_pose_deviation'].func is mdp.joint_pose_deviation_cost
    assert cfg.rewards['joint_pose_deviation'].weight == -.01
    assert 'joint_pose_deviation' not in cfg.terminations
    assert cfg.metrics['joint_pose_rms_rad'].func is mdp.joint_pose_rms
    env.scene['robot'].data.joint_pos[:] = .3
    before = mdp.joint_pose_deviation_cost(env).clone()
    env.scene['robot'].data.body_link_quat_w[:] = torch.tensor([0., 0., 0., 1.])
    torch.testing.assert_close(mdp.joint_pose_deviation_cost(env), before)
