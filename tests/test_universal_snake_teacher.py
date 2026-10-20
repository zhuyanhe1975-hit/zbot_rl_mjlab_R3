"""Optional archived snake teacher: native semantics, routing and memory isolation."""

import math
from types import SimpleNamespace

import pytest
import test_universal_teachers as teacher_fixtures
import torch

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.teachers import SNAKE_CHECKPOINT, TeacherEnsemble

env = teacher_fixtures.env


@pytest.fixture
def snake_env(env):
    data = env.scene['robot'].data
    data.joint_pos.zero_()
    quat = torch.tensor([math.sqrt(.5), 0., -math.sqrt(.5), 0.])
    data.body_link_quat_w[:] = quat
    data.body_link_pos_w[:, :, 2] = .0555
    data.body_com_lin_vel_w = data.body_link_lin_vel_w.clone()
    data.site_quat_w[:] = quat
    data.site_pos_w[:, 0, 0] = .5
    data.site_ang_vel_w = torch.full((6, 2, 3), .15)
    robot = env.scene['robot']
    robot.find_sites = lambda name, preserve_order=True: (([0, 1], name) if not isinstance(name, str) else ([0 if name == 'foot_0_sole' else 1], [name]))
    robot.indexing = SimpleNamespace(body_ids=torch.tensor([0]))
    env.sim = SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(6, 1)))
    action = env.action_manager.get_term('joint_pos')
    action.commanded_target.zero_()
    command = env.command_manager.get_term('transform')
    command.standing_mode[:] = False
    command.transition_mask[:] = False
    command.requested_velocity[:] = torch.tensor([.08, 0., 0.])
    command.velocity_command[:] = command.requested_velocity
    command.heading = torch.tensor([1., 0.]).expand(6, 2).clone()
    return env


@pytest.fixture
def snake_teachers(snake_env):
    return TeacherEnsemble(snake_env, checkpoints={'snake': SNAKE_CHECKPOINT})


def test_optional_teacher_keeps_default_set(env, snake_teachers):
    assert TeacherEnsemble(env).teacher_names == ('walking', 'omni', 'transform')
    assert snake_teachers.teacher_names == ('walking', 'omni', 'transform', 'snake')
    model = snake_teachers.models['snake']
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    checkpoint = torch.load(SNAKE_CHECKPOINT, map_location='cpu', weights_only=False)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint['actor_state_dict'][name])


def test_snake_observation_matches_reference_without_student_memory(snake_env, snake_teachers):
    before = dict(vars(snake_env))
    action = snake_env.action_manager.get_term('joint_pos')
    action.rate_cfg.max_target_speed = math.pi
    observation = snake_teachers.observation('snake')
    assert observation.shape == (6, 51)
    proxy = snake_teachers._snake_proxy
    expected = torch.tensor(
        [0.] * 6 + [.2] * 6 + [0.] * 6 + [.1] * 6
        + [0., 0., -1.] + [.1, .1, .15] + [-1., 0., 0.]
        + [.2, .2, -.2] + [.0555, .1] + [0., 1.]
        + [.1, .1, -.1] + [math.sqrt(.5), 0., -math.sqrt(.5), 0.]
        + [0., 1., 0., 0.]
    ).expand(6, 51)
    torch.testing.assert_close(observation, expected)
    torch.testing.assert_close(observation[:, 18:24], action.raw_action / 2)
    # World +X expressed in the archived head frame; not a velocity command.
    torch.testing.assert_close(observation[:, 24:27], torch.tensor([0., 0., -1.]).expand(6, 3))
    torch.testing.assert_close(observation[:, 47:49], torch.tensor([0., 1.]).expand(6, 2))
    assert vars(snake_env).keys() == before.keys()
    assert not any(name.startswith('_snake') for name in vars(snake_env))
    same = snake_teachers.observation('snake')
    torch.testing.assert_close(observation, same)
    snake_env.common_step_counter += 1
    snake_env.episode_length_buf += 1
    mdp.head_yaw_integral_state(proxy).value.fill_(.2)
    next_observation = snake_teachers.observation('snake')
    assert (next_observation[:, -1] > 0).all()
    snake_env.common_step_counter += 1
    snake_env.episode_length_buf.zero_()
    reset = snake_teachers.observation('snake')
    torch.testing.assert_close(reset[:, -2:], torch.zeros(6, 2))


def test_snake_rate_conversion_uses_native_speed_and_bounds(snake_env, snake_teachers):
    action = snake_env.action_manager.get_term('joint_pos')
    action.rate_cfg.max_target_speed = math.pi
    output = torch.full((6, 6), .25)
    torch.testing.assert_close(snake_teachers._convert('snake', output), torch.full((6, 6), .5))
    native_limit = snake_teachers.configs['snake'].actions['joint_pos'].target_limit
    action.commanded_target.fill_(native_limit - .01)
    expected = torch.full((6, 6), .01 / (snake_env.step_dt * math.pi))
    torch.testing.assert_close(snake_teachers._convert('snake', torch.ones(6, 6)), expected)


def test_snake_only_routes_stable_forward_world_x(snake_env, snake_teachers):
    command = snake_env.command_manager.get_term('transform')
    command.requested_velocity[1] = torch.tensor([-.08, 0., 0.])
    command.requested_velocity[2] = torch.tensor([0., .08, 0.])
    command.requested_velocity[3] = torch.tensor([.08, 0., .1])
    command.transition_mask[4] = True
    command.heading[5] = torch.tensor([0., 1.])
    command.velocity_command[:] = command.requested_velocity
    one, two = snake_teachers.query(), snake_teachers.query()
    assert one.counts['snake'] == 1
    assert one.teacher_ids.tolist() == [3, -1, -1, -1, 2, -1]
    torch.testing.assert_close(one.actions, two.actions)
    assert torch.isfinite(one.actions).all()
    assert one.actions.abs().max() <= 1


def test_snake_does_not_supervise_outside_posture_or_target_coverage(snake_env, snake_teachers):
    action = snake_env.action_manager.get_term('joint_pos')
    action.commanded_target[0, 0] = math.pi / 3 + .01
    snake_env.scene['robot'].data.joint_pos[1, 0] = math.pi / 3 + .11
    snake_env.scene['robot'].data.body_link_pos_w[2, 0, 2] = .2
    result = snake_teachers.query()
    assert result.teacher_ids[:3].tolist() == [-1, -1, -1]
    assert result.counts['snake'] == 3


def test_snake_checkpoint_recipe_and_state_are_strict(snake_env, tmp_path):
    checkpoint = torch.load(SNAKE_CHECKPOINT, map_location='cpu', weights_only=False)
    checkpoint['actor_state_dict']['unexpected.weight'] = torch.zeros(1)
    bad = tmp_path / 'bad.pt'
    torch.save(checkpoint, bad)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        TeacherEnsemble(snake_env, checkpoints={'snake': str(bad)})
    checkpoint['infos']['snake_recipe']['version'] = -1
    torch.save(checkpoint, bad)
    with pytest.raises(ValueError, match='第22版'):
        TeacherEnsemble(snake_env, checkpoints={'snake': str(bad)})


@pytest.mark.parametrize(('degrees', 'count'), [(20., 6), (65., 0)])
def test_snake_native_head_yaw_envelope(snake_env, degrees, count):
    # Legitimate head oscillation remains covered, while initial +X-facing
    # domain is still enforced independently of episode-relative yaw history.
    theta = math.radians(degrees)
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    r = math.sqrt(.5)
    quat = torch.tensor([r * c, r * s, -r * c, r * s])
    data = snake_env.scene['robot'].data
    data.body_link_quat_w[:] = quat
    data.site_quat_w[:] = quat
    data.site_pos_w[:, 0, :2] = torch.tensor([math.cos(theta), math.sin(theta)]) * .5
    teachers = TeacherEnsemble(snake_env, checkpoints={'snake': SNAKE_CHECKPOINT})
    labels = teachers.query()
    assert labels.counts['snake'] == count


def test_snake_accepts_native_servo_overshoot_without_widening_target(snake_env, snake_teachers):
    snake_env.scene['robot'].data.joint_pos[:, 0] = math.pi / 3 + .05
    labels = snake_teachers.query()
    assert labels.counts['snake'] == 6
    action = snake_env.action_manager.get_term('joint_pos')
    action.commanded_target[:, 0] = math.pi / 3 + .001
    assert snake_teachers.query().counts['snake'] == 0
