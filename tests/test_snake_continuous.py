"""蛇形连续探索、身体运动坐标与旧模型观察语义的边界。"""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.commands import SnakeVelocityCommandCfg
from zbot_rl_mjlab.tasks.snake.env_cfg import SNAKE_JOINT_LIMIT, snake_env_cfg


def continuous_env():
    env = state_env()
    env.scene['robot'].find_bodies = lambda name: ([6 if name == 'base' else 0], [name])
    return env


def yaw_quat(angle):
    return torch.tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])


@pytest.mark.parametrize('yaw', [0., .3, math.pi / 2, -2.])
def test_motion_axes_construct_left_then_forward_from_head_z_and_gravity(yaw):
    env = continuous_env()
    data = env.scene['robot'].data
    q = quat_mul(yaw_quat(yaw).expand(2, -1), data.body_link_quat_w[:, 0])
    data.body_link_quat_w[:, 0] = q
    forward, left = mdp.motion_axes(env)
    up = torch.tensor([[0., 0., 1.]]).expand(2, -1)
    body_z = quat_apply(q, up)
    expected_left = torch.linalg.cross(-up, body_z)
    expected_left /= expected_left.norm(dim=-1, keepdim=True)
    expected_forward = torch.linalg.cross(expected_left, up)
    torch.testing.assert_close(forward, expected_forward[:, :2])
    torch.testing.assert_close(left, expected_left[:, :2])
    forward_w = torch.cat((forward, torch.zeros(2, 1)), -1)
    left_w = torch.cat((left, torch.zeros(2, 1)), -1)
    torch.testing.assert_close(torch.linalg.cross(forward_w, left_w), up)
    torch.testing.assert_close((forward * left).sum(-1), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(forward.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(left, torch.stack((-forward[:, 1], forward[:, 0]), -1))


def test_base_orientation_and_yaw_do_not_change_head_motion_frame():
    env = continuous_env()
    data = env.scene['robot'].data
    initial = mdp.head_forward(env).clone()
    data.body_link_quat_w[:, 6] = quat_mul(yaw_quat(math.pi / 2).expand(2, -1), data.body_link_quat_w[:, 6])
    torch.testing.assert_close(mdp.head_forward(env), initial)
    data.body_com_lin_vel_w[..., 0] = .05
    data.body_link_ang_vel_w[:, 0, 2] = -.4
    data.body_link_ang_vel_w[:, 6, 2] = .3
    torch.testing.assert_close(mdp.planar_velocity(env), torch.tensor([[.05, 0., -.4]]).expand(2, -1))


@pytest.mark.parametrize('yaw', [0., math.pi / 2, -math.pi / 2])
def test_legacy_visualized_command_rotates_with_foot_0_independently_of_base(yaw):
    env = continuous_env()
    env.cfg.observation_reference = 'command'
    data = env.scene['robot'].data
    data.body_link_quat_w[:, 0] = quat_mul(
        yaw_quat(yaw).expand(2, -1), data.body_link_quat_w[:, 0],
    )
    term = SnakeVelocityCommandCfg(resampling_time_range=(4., 8.), debug_vis=True).build(env)
    term._command[:, 0] = .25
    arrows = []
    visualizer = SimpleNamespace(
        get_env_indices=lambda count: [0],
        add_sphere=lambda *args, **kwargs: None,
        add_arrow=lambda start, end, **kwargs: arrows.append(torch.as_tensor(end - start)),
    )
    expected = torch.tensor([math.cos(yaw), math.sin(yaw), 0.]) * .15
    term.debug_vis(visualizer)
    torch.testing.assert_close(arrows[-1], expected, atol=1e-6, rtol=0)
    data.body_link_quat_w[:, 6] = torch.tensor([1., 0., 0., 0.])
    term.debug_vis(visualizer)
    torch.testing.assert_close(arrows[-1], expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize('play', [False, True])
def test_legacy_tracking_and_visual_share_site_pose_instead_of_body_pose(play):
    env = continuous_env()
    env.cfg = snake_env_cfg(play=play)
    env.cfg.observation_reference = 'command'
    data = env.scene['robot'].data
    # site独立转90度且偏离刚体原点，检测是否误读body姿态或添加显示偏移。
    data.site_quat_w = data.site_quat_w.clone()
    data.site_quat_w[:, 0] = quat_mul(
        yaw_quat(math.pi / 2).expand(2, -1), data.site_quat_w[:, 0],
    )
    data.site_pos_w = torch.tensor([[[2., 3., .5]], [[4., 5., .7]]])
    data.body_com_lin_vel_w[..., 1] = .05
    data.site_ang_vel_w = data.site_ang_vel_w.clone()
    data.site_ang_vel_w[:, 0, 2] = .2
    data.body_link_ang_vel_w[:, 0, 2] = -.4
    data.body_link_ang_vel_w[:, 6, 2] = -.6
    env.scene['snake_ground_contact'].data.force[..., 2] = 9.81
    term = mdp.command_term(env)
    term.command[:] = torch.tensor([.05, 0., .2])
    term.requested_velocity[:] = torch.tensor([.05, 0., 0.])
    torch.testing.assert_close(
        mdp.actor_state(env)[:, 27:30], term.command, atol=1e-6, rtol=0,
    )
    # 平移与yaw独立跟踪，命中平移目标时指数奖励为1。
    torch.testing.assert_close(mdp.translation_tracking(env), torch.ones(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(mdp.velocity_error(env), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(mdp.direction_motion_metric(env, 0, 1), torch.full((2,), .05))
    visual_term = SnakeVelocityCommandCfg(resampling_time_range=(4., 8.), debug_vis=True).build(env)
    visual_term._command.copy_(term.command)
    arrows = []
    markers = []
    visual_term.debug_vis(SimpleNamespace(
        get_env_indices=lambda count: [0],
        add_sphere=lambda center, **kwargs: markers.append(torch.as_tensor(center)),
        add_arrow=lambda start, end, **kwargs: arrows.append((torch.as_tensor(start), torch.as_tensor(end))),
    ))
    origin = data.site_pos_w[0, 0]
    torch.testing.assert_close(markers[0], origin)
    torch.testing.assert_close(arrows[0][0], origin)
    torch.testing.assert_close(arrows[0][1], origin + torch.tensor([0., .04, 0.]))


def test_degenerate_body_z_retains_last_valid_heading_per_environment():
    env = continuous_env()
    data = env.scene['robot'].data
    data.body_link_quat_w[1, 0] = quat_mul(yaw_quat(math.pi / 2), data.body_link_quat_w[1, 0])
    expected = mdp.head_forward(env).clone()
    data.body_link_quat_w[:, 0] = torch.tensor([1., 0., 0., 0.])
    torch.testing.assert_close(mdp.head_forward(env), expected)
    assert torch.isfinite(mdp.actor_state(env)).all()


def test_initial_degenerate_heading_uses_body_axis_and_rotates_with_body_yaw():
    env = continuous_env()
    env.scene['robot'].data.body_link_quat_w[:, 0] = torch.stack((yaw_quat(0.), yaw_quat(math.pi / 2)))
    torch.testing.assert_close(mdp.head_forward(env), torch.tensor([[1., 0.], [0., 1.]]), atol=1e-6, rtol=0)


def test_partial_reset_initializes_only_reset_environment_heading():
    env = continuous_env()
    robot = env.scene['robot']
    robot.data.default_root_state = torch.zeros(2, 13)
    robot.data.default_root_state[:, 3] = 1
    robot.write_root_state_to_sim = lambda states, env_ids: None
    robot.write_joint_state_to_sim = lambda pos, vel, env_ids: None
    mdp.command_term(env).cfg.training = False
    env.common_step_counter = 0
    env._snake_forward_w = torch.tensor([[0., -1.], [0., 1.]])
    mdp.reset_snake(env, torch.tensor([0]))
    torch.testing.assert_close(env._snake_forward_w, torch.tensor([[1., 0.], [0., 1.]]), atol=1e-6, rtol=0)


@pytest.mark.parametrize('play', [False, True])
@pytest.mark.parametrize('random_yaw', [0., math.pi / 2, math.pi])
def test_reset_uses_rotated_initial_pose_and_preserves_training_yaw_sampling(monkeypatch, play, random_yaw):
    env = continuous_env()
    env.cfg = snake_env_cfg(play=play)
    robot = env.scene['robot']
    robot.data.default_root_state = torch.zeros(2, 13)
    robot.data.default_root_state[:, 3] = 1
    roots = []
    robot.write_root_state_to_sim = lambda states, env_ids: roots.append(states.clone())
    robot.write_joint_state_to_sim = lambda pos, vel, env_ids: None
    mdp.command_term(env).cfg.training = not play
    monkeypatch.setattr(torch, 'rand', lambda count, device: torch.full((count,), random_yaw / (2 * math.pi)))
    mdp.reset_snake(env, torch.tensor([0]))
    initial = torch.tensor([math.sqrt(.5), 0., -math.sqrt(.5), 0.])
    torch.testing.assert_close(torch.tensor(env.cfg.scene.entities['robot'].init_state.rot), initial)
    expected = quat_mul(yaw_quat(0. if play else random_yaw), initial)
    torch.testing.assert_close(roots[0][0, 3:7], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(roots[0][0, :3], torch.tensor([0., 0., .0555]))
    assert roots[0][0, 7:].eq(0).all()


def test_actor_world_x_reference_changes_under_yaw_but_body_features_are_invariant():
    env = continuous_env()
    data = env.scene['robot'].data
    data.body_link_ang_vel_w[:] = torch.tensor([.4, -.3, .25])
    data.body_com_lin_vel_w[:] = torch.tensor([.04, .03, .02])
    data.body_link_lin_vel_w[:] = torch.tensor([.04, .03, .02])
    q = yaw_quat(math.pi / 2)
    for name in ('body_link_ang_vel_w', 'body_com_lin_vel_w', 'body_link_lin_vel_w', 'body_link_pos_w', 'body_com_pos_w'):
        value = getattr(data, name)
        value[1] = quat_apply(q.expand(12, -1), value[1])
    data.body_link_quat_w[1] = quat_mul(q.expand(12, -1), data.body_link_quat_w[1])
    obs = mdp.actor_state(env)
    invariant = torch.cat((obs[:, :24], obs[:, 27:]), -1)
    torch.testing.assert_close(invariant[0], invariant[1], atol=2e-6, rtol=1e-5)
    assert not torch.allclose(obs[0, 24:27], obs[1, 24:27])
    world_reference = quat_apply(data.body_link_quat_w[:, 0], obs[:, 24:27])
    torch.testing.assert_close(world_reference, torch.tensor([[1., 0., 0.]]).expand(2, -1), atol=1e-6, rtol=0)
    expected = quat_apply_inverse(data.body_link_quat_w[:, 0], data.body_link_ang_vel_w[:, 0])
    torch.testing.assert_close(obs[:, 33:36], expected)
    assert obs.shape == (2, 40)
    assert mdp.critic_state(env).shape == (2, 55)


def test_legacy_head_motion_and_world_gyro_keep_old_checkpoint_semantics():
    env = continuous_env()
    env.cfg.motion_frame = 'legacy_head'
    env.cfg.head_gyro_frame = 'world'
    data = env.scene['robot'].data
    data.body_com_lin_vel_w[..., 0] = .05
    data.body_link_ang_vel_w[:, 0] = torch.tensor([.4, -.3, -.25])
    data.body_link_ang_vel_w[:, 6, 2] = .6
    torch.testing.assert_close(mdp.head_forward(env), torch.tensor([[1., 0.], [1., 0.]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(mdp.planar_velocity(env), torch.tensor([[.05, 0., -.25]]).expand(2, -1), atol=1e-6, rtol=0)
    torch.testing.assert_close(mdp.actor_state(env)[:, 33:36], data.body_link_ang_vel_w[:, 0])
    data.body_link_quat_w[:, 0] = torch.tensor([1., 0., 0., 0.])
    torch.testing.assert_close(mdp.head_forward(env), torch.tensor([[1., 0.], [1., 0.]]))


def test_large_head_tilt_resets_without_disabling_supported_motion_reward():
    env = continuous_env()
    data = env.scene['robot'].data
    data.body_link_quat_w[:, 0] = torch.tensor([1., 0., 0., 0.])
    data.body_com_pos_w[..., 2] = .4
    data.body_com_lin_vel_w[..., 0] = .05
    mdp.command_term(env).command[:, 0] = .05
    env.scene['snake_ground_contact'].data.force[..., 2] = 9.81
    assert not mdp.unsafe(env).any()
    assert mdp.head_tilt_limit_reached(env).all()
    assert mdp.translation_tracking(env).min() > .99
    assert mdp.head_tilt_cost(env).tolist() == pytest.approx([.5, .5])


@pytest.mark.parametrize('danger', ['nonfinite', 'workspace', 'impact'])
def test_real_safety_failures_still_terminate(danger):
    env = continuous_env()
    if danger == 'nonfinite':
        env.scene['robot'].data.joint_vel[0, 0] = float('nan')
    elif danger == 'workspace':
        env.scene['robot'].data.root_link_pos_w[0, 0] = env.cfg.workspace_radius + 1
    else:
        env.scene['snake_ground_contact'].data.force_history[0, 0, 0, 2] = 9 * 12 * 9.81
    assert mdp.unsafe(env).tolist() == [True, False]


def test_self_contact_keeps_termination_without_auxiliary_failure_penalty():
    env = continuous_env()
    env.scene['snake_self_contact'].data.found[0, 0] = 1
    assert mdp.self_collision(env).tolist() == [True, False]
    assert env.cfg.terminations['self_collision'].func is mdp.self_collision
    assert 'failure' not in env.cfg.rewards


def test_camera_tilt_cost_is_bounded_and_zero_when_level():
    env = continuous_env()
    torch.testing.assert_close(mdp.head_tilt_cost(env), torch.zeros(2), atol=1e-6, rtol=0)
    data = env.scene['robot'].data
    data.body_link_quat_w[:, 0] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
    torch.testing.assert_close(mdp.head_tilt_cost(env), torch.full((2,), .5))


def test_world_speed_rewards_preserve_physical_action_bounds():
    cfg = snake_env_cfg()
    assert cfg.motion_frame == 'body_gravity'
    assert cfg.head_gyro_frame == 'head'
    assert {name: term.weight for name, term in cfg.rewards.items()} == {
        'forward_speed': 1., 'lateral_speed_cost': -1., 'vertical_speed_cost': -1.,
        'joint_pose_deviation': -.01, 'head_tilt_reset_cost': -.5, 'target_saturation_fraction': -.1,
    }
    assert cfg.actions['joint_pos'].target_limit == SNAKE_JOINT_LIMIT == math.pi / 2
    assert cfg.actions['joint_pos'].max_target_speed == 2 * math.pi
    assert 'translation_stall_fraction' not in cfg.metrics


def test_target_saturation_adds_penalty_but_action_clipping_is_only_a_metric():
    env = continuous_env()
    action = env.action_manager.get_term('joint_pos')
    action.commanded_target[0, :3] = math.pi / 2
    action.raw_action[1, :2] = torch.tensor([-1., 1.])
    assert mdp.target_saturation_fraction(env).tolist() == pytest.approx([.5, 0.])
    assert mdp.action_clip_fraction(env).tolist() == pytest.approx([0., 1 / 3])
    assert env.cfg.rewards['target_saturation_fraction'].weight == -.1
    assert 'action_clip_fraction' not in env.cfg.rewards


def test_target_windup_only_penalizes_ineffective_outward_requests():
    env = continuous_env()
    action = env.action_manager.get_term('joint_pos')
    action.commanded_target[:] = torch.tensor([math.pi / 2, -math.pi / 2, .5, -.5, 0., 0.])
    env.action_manager.action = torch.tensor([[2., -2., 2., -2., 2., -2.], [-2., 2., 2., -2., 2., -2.]])
    # 原始输出2不能被执行层的±1裁剪掩盖；反向脱离边界没有惩罚。
    assert mdp.target_windup_cost(env).tolist() == pytest.approx([8 / 6, 0.])
    env.action_manager.action[0, :2] = torch.tensor([1., -1.])
    assert mdp.target_windup_cost(env)[0].item() == pytest.approx(2 / 6)
    assert 'target_windup' not in env.cfg.rewards


def test_velocity_error_uses_current_command_ranges(monkeypatch):
    env = continuous_env()
    term = mdp.command_term(env)
    term.command.zero_()
    actual = torch.tensor([[.25, 0., 0.], [0., 0., math.pi / 2]])
    monkeypatch.setattr(mdp, 'planar_velocity', lambda env: actual)
    torch.testing.assert_close(mdp.velocity_error(env), torch.full((2,), math.sqrt(1 / 3)))
