"""New guidance has explicit task domains and an isolated yaw-canonical view."""

import math

import pytest
import test_universal_snake_teacher as snake_fixtures
import test_universal_teachers as teacher_fixtures
import torch
from mjlab.utils.lab_api.math import quat_mul

from zbot_rl_mjlab.guided_teachers import GuidedTeacherEnsemble
from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.teachers import DEFAULT_CHECKPOINTS, SNAKE_CHECKPOINT

env = teacher_fixtures.env
snake_env = snake_fixtures.snake_env


@pytest.fixture
def guided_env(snake_env):
    command = snake_env.command_manager.get_term('transform')
    command.current_task = torch.full((6,), 2, dtype=torch.long)
    return snake_env


@pytest.fixture
def teachers(guided_env):
    return GuidedTeacherEnsemble(guided_env, include_omni=False, include_snake=True)


def test_loads_only_exact_frozen_archives(guided_env, teachers):
    assert teachers.teacher_names == ('transform', 'snake')
    for name, path in (('transform', DEFAULT_CHECKPOINTS['transform']), ('snake', SNAKE_CHECKPOINT)):
        model = teachers.models[name]
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        saved = torch.load(path, map_location='cpu', weights_only=False)['actor_state_dict']
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved[key])
    with pytest.raises(ValueError, match='未启用'):
        GuidedTeacherEnsemble(guided_env, checkpoints={'walking': DEFAULT_CHECKPOINTS['walking']})


def test_never_guides_quiet_biped_or_quiet_snake(guided_env, teachers):
    command = guided_env.command_manager.get_term('transform')
    command.current_task[:] = torch.tensor([0, 2, 1, 3, 0, 2])
    command.requested_velocity.zero_()
    command.velocity_command.zero_()
    command.transition_mask[:] = torch.tensor([False, False, True, True, False, False])
    command.standing_mode[:] = command.current_task == 3
    result = teachers.query()
    assert result.teacher_ids.tolist() == [-1, -1, 0, 0, -1, -1]
    assert result.counts == {'transform': 2, 'snake': 0}


def test_snake_domains_are_explicit_and_finite(guided_env, teachers):
    command = guided_env.command_manager.get_term('transform')
    command.requested_velocity[1] = torch.tensor([-.08, 0., 0.])
    command.requested_velocity[2] = torch.tensor([0., .08, 0.])
    command.requested_velocity[3] = torch.tensor([.08, 0., .1])
    command.velocity_command[4] = torch.tensor([-.08, 0., 0.])
    guided_env.scene['robot'].data.body_link_ang_vel_w[5, 0, 0] = torch.nan
    result = teachers.query()
    assert result.teacher_ids.tolist() == [1, -1, -1, -1, -1, -1]
    assert torch.isfinite(result.actions).all() and result.actions.abs().max() <= 1


def rotate_world(env, yaw):
    data = env.scene['robot'].data
    rotation = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    for name in ('body_link_quat_w', 'site_quat_w'):
        value = getattr(data, name)
        value.copy_(quat_mul(rotation.expand_as(value), value))
    for name in ('body_link_pos_w', 'site_pos_w', 'body_link_lin_vel_w', 'body_link_ang_vel_w', 'body_com_lin_vel_w', 'site_ang_vel_w'):
        value = getattr(data, name)
        x, y = value[..., 0].clone(), value[..., 1].clone()
        value[..., 0] = math.cos(yaw) * x - math.sin(yaw) * y
        value[..., 1] = math.sin(yaw) * x + math.cos(yaw) * y
    command = env.command_manager.get_term('transform')
    command.heading[:] = torch.tensor([math.cos(yaw), math.sin(yaw)])


@pytest.mark.parametrize('yaw', [math.pi / 2, -math.pi / 2, math.pi, .718])
def test_random_global_yaw_equivariance_and_no_mutation(guided_env, teachers, yaw):
    reference = teachers.query()
    observation = teachers._canonical_snake_observation(
        guided_env.command_manager.get_term('transform').current_task,
        torch.ones(6, dtype=torch.bool),
    ).clone()
    rotate_world(guided_env, yaw)
    rotated = GuidedTeacherEnsemble(guided_env, include_omni=False, include_snake=True)
    snapshots = {name: value.clone() for name, value in vars(guided_env.scene['robot'].data).items() if isinstance(value, torch.Tensor)}
    env_keys = set(vars(guided_env))
    actual = rotated.query()
    assert actual.counts['snake'] == 6
    torch.testing.assert_close(actual.actions, reference.actions, atol=1e-5, rtol=1e-4)
    transformed = rotated._canonical_snake_observation(
        guided_env.command_manager.get_term('transform').current_task,
        torch.ones(6, dtype=torch.bool),
    )
    torch.testing.assert_close(transformed, observation, atol=1e-6, rtol=1e-5)
    assert set(vars(guided_env)) == env_keys
    for name, value in snapshots.items():
        torch.testing.assert_close(getattr(guided_env.scene['robot'].data, name), value)
    again = rotated.query()
    torch.testing.assert_close(again.actions, actual.actions)
    assert not hasattr(guided_env, '_snake_initial_yaw')


def test_partial_resets_and_task_entry_reset_only_proxy_history(guided_env, teachers):
    teachers.query()
    proxy = teachers._guided_proxy
    body = mdp.body_yaw_integral_state(proxy)
    head = mdp.head_yaw_integral_state(proxy)
    guided_env.episode_length_buf.fill_(10)
    guided_env.common_step_counter = 10
    teachers.query()
    body.value.fill_(.3)
    head.value.fill_(.4)
    guided_env.episode_length_buf[2] = 0
    guided_env.common_step_counter = 11
    teachers.query()
    assert body.value[2] == head.value[2] == 0
    torch.testing.assert_close(body.value[[0, 1, 3, 4, 5]], torch.full((5,), .3))
    command = guided_env.command_manager.get_term('transform')
    command.current_task[3] = 0
    teachers.query()
    body.value[3] = .7
    command.current_task[3] = 2
    teachers.query()
    assert body.value[3] == 0
    assert not hasattr(guided_env, '_snake_body_yaw_integral')


def test_joint_limits_and_transform_asymmetry_reject_labels(guided_env, teachers):
    action = guided_env.action_manager.get_term('joint_pos')
    action.commanded_target[0, 0] = math.pi / 3 + .001
    guided_env.scene['robot'].data.joint_pos[1, 0] = math.pi / 3 + .101
    command = guided_env.command_manager.get_term('transform')
    command.current_task[2:4] = 3
    command.transition_mask[2:4] = True
    action.commanded_target[2, 0] = 2.
    guided_env.scene['robot'].data.joint_pos[3, 0] = 2.
    result = teachers.query()
    assert result.teacher_ids.tolist() == [-1, -1, -1, -1, 1, 1]


def test_zero_yaw_51_columns_match_archive_reference(guided_env, teachers):
    from zbot_rl_mjlab.teachers import TeacherEnsemble

    reference = TeacherEnsemble(guided_env, enabled_teachers=('snake',))
    actual = teachers._canonical_snake_observation(
        guided_env.command_manager.get_term('transform').current_task,
        torch.ones(6, dtype=torch.bool),
    )
    torch.testing.assert_close(actual, reference.observation('snake'))


def test_late_task_entry_shares_policy_visible_episode_phase(guided_env, teachers):
    from zbot_rl_mjlab.tasks.universal import mdp as universal_mdp

    command = guided_env.command_manager.get_term('transform')
    command.gait_frequency = torch.full((6,), .5)
    guided_env.cfg.snake_teacher_enabled = True
    command.current_task.zero_()
    teachers.query()
    guided_env.episode_length_buf[:] = torch.tensor([113, 127, 139, 151, 163, 179])
    guided_env.common_step_counter = 179
    command.current_task[2] = 2
    teachers.query()
    actual = teachers._canonical_snake_observation(
        command.current_task, command.current_task == 2,
    )
    phase = guided_env.episode_length_buf * guided_env.step_dt * .7 * (2 * torch.pi)
    expected = torch.stack((phase.sin(), phase.cos()), -1)
    torch.testing.assert_close(actual[:, 38:40], expected)
    torch.testing.assert_close(actual[:, 38:40], universal_mdp.gait_observation(guided_env)[:, :2])
    assert teachers._guided_proxy.episode_length_buf is guided_env.episode_length_buf
    # A different selected robot entering snake mode keeps the same shared clock.
    command.current_task[4] = 2
    later = teachers._canonical_snake_observation(
        command.current_task, command.current_task == 2,
    )
    torch.testing.assert_close(later[:, 38:40], expected)


@pytest.fixture
def omni_env(guided_env):
    data = guided_env.scene['robot'].data
    nominal = torch.tensor([.312, .837, -2.02, 2.02, -.837, -.312]).expand(6, 6).clone()
    data.default_joint_pos = nominal
    data.joint_pos.copy_(nominal)
    data.body_link_quat_w[:] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
    data.body_link_pos_w[:, :, 2] = .264
    action = guided_env.action_manager.get_term('joint_pos')
    action.commanded_target.copy_(nominal)
    action.rate_cfg.biped_residual_scale = .375
    command = guided_env.command_manager.get_term('transform')
    command.current_task.zero_()
    command.standing_mode.fill_(True)
    command.transition_mask.zero_()
    command.initial_heading = command.heading.clone()
    command.gait_frequency = torch.full((6,), .7)
    return guided_env


def test_default_omni_archive_is_exact_and_frozen(omni_env):
    teachers = GuidedTeacherEnsemble(omni_env)
    assert teachers.teacher_names == ('omni', 'transform')
    model = teachers.models['omni']
    saved = torch.load(DEFAULT_CHECKPOINTS['omni'], map_location='cpu', weights_only=False)['actor_state_dict']
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name])
    assert teachers.observation('omni').shape == (6, 33)
    with pytest.raises(ValueError, match='未启用'):
        GuidedTeacherEnsemble(omni_env, checkpoints={'walking': DEFAULT_CHECKPOINTS['walking']})


def test_omni_task_zero_covers_stop_backward_side_turn_and_combined(omni_env):
    teachers = GuidedTeacherEnsemble(omni_env)
    command = omni_env.command_manager.get_term('transform')
    command.requested_velocity[:] = torch.tensor([
        [0., 0., 0.], [-.1, 0., 0.], [0., .05, 0.], [0., 0., .3],
        [.1, -.05, -.3], [.08, .03, .2],
    ])
    command.velocity_command.copy_(command.requested_velocity)
    result = teachers.query()
    assert result.counts == {'omni': 6, 'transform': 0}
    assert result.teacher_ids is not None
    assert result.teacher_ids.tolist() == [0] * 6
    assert torch.isfinite(result.actions).all() and result.actions.abs().max() <= 1
    command.current_task[:] = torch.tensor([0, 1, 2, 3, 0, 0])
    command.transition_mask[1:4] = True
    omni_env.scene['robot'].data.body_link_pos_w[4, 0, 2] = .08
    omni_env.scene['robot'].data.joint_pos[5, 0] += 2.
    result = teachers.query()
    assert result.teacher_ids is not None
    assert result.teacher_ids[0] == 0
    assert (result.teacher_ids[1:] != 0).all()
    assert result.counts['omni'] == 1


def test_omni_inverse_uses_per_env_nominal_and_absolute_position_bounds(omni_env):
    teachers = GuidedTeacherEnsemble(omni_env)
    data = omni_env.scene['robot'].data
    action = omni_env.action_manager.get_term('joint_pos')
    data.default_joint_pos += torch.linspace(-.03, .03, 6)[:, None]
    output = torch.linspace(-4., 4., 36).reshape(6, 6)
    nominal = data.default_joint_pos
    desired = teachers.defaults['omni'] + .25 * output
    desired = desired.clamp(min=nominal - .375, max=nominal + .375)
    limits = action._target_limits()
    limits[:, 0, 1] = nominal[:, 0] + .1
    desired = desired.clamp(min=limits[..., 0], max=limits[..., 1])
    converted = teachers._convert('omni', output)
    torch.testing.assert_close(nominal + .375 * converted, desired)
    # A velocity inverse would depend on the previous integrated target. This must not.
    action.commanded_target.add_(.1)
    torch.testing.assert_close(teachers._convert('omni', output), converted)
    subset = torch.tensor([1, 4])
    torch.testing.assert_close(teachers._convert('omni', output[subset], subset), converted[subset])


def test_omni_fallback_nominal_and_last_action_semantics(omni_env):
    teachers = GuidedTeacherEnsemble(omni_env)
    data = omni_env.scene['robot'].data
    del data.default_joint_pos
    output = torch.full((6, 6), .3)
    torch.testing.assert_close(teachers._convert('omni', output), output * (.25 / .375))
    action = omni_env.action_manager.get_term('joint_pos')
    action.commanded_target.copy_(teachers.defaults['omni'] + .25 * output)
    torch.testing.assert_close(teachers.observation('omni')[:, 21:27], output)


@pytest.mark.parametrize('yaw', [math.pi / 2, -.718, math.pi])
def test_omni_observation_frame_and_clock_are_visible_and_yaw_invariant(omni_env, yaw):
    teachers = GuidedTeacherEnsemble(omni_env)
    command = omni_env.command_manager.get_term('transform')
    command.gait_frequency[:] = torch.linspace(.3, .9, 6)
    omni_env.episode_length_buf[:] = torch.arange(6) * 17 + 11
    command.velocity_command[:] = torch.tensor([.1, .02, .3])
    command.requested_velocity.copy_(command.velocity_command)
    before_data = {name: value.clone() for name, value in vars(omni_env.scene['robot'].data).items() if isinstance(value, torch.Tensor)}
    before_command = {name: value.clone() for name, value in vars(command).items() if isinstance(value, torch.Tensor)}
    reference = teachers.observation('omni')
    torch.testing.assert_close(reference[:, -3:], torch.tensor([-.1, -.02, .3]).expand(6, 3))
    expected_phase = omni_env.episode_length_buf * omni_env.step_dt * command.gait_frequency * (2 * torch.pi)
    torch.testing.assert_close(reference[:, 27:29], torch.stack((expected_phase.sin(), expected_phase.cos()), -1))
    for name, value in before_data.items():
        torch.testing.assert_close(getattr(omni_env.scene['robot'].data, name), value)
    for name, value in before_command.items():
        torch.testing.assert_close(getattr(command, name), value)
    rotate_world(omni_env, yaw)
    command.initial_heading[:] = command.heading
    actual = GuidedTeacherEnsemble(omni_env).observation('omni')
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)


def test_omni_conversion_matches_actual_hybrid_action_with_slew_limit(omni_env, monkeypatch):
    from mjlab.envs.mdp.actions import JointPositionAction

    from zbot_rl_mjlab.tasks.universal.actions import (
        SnakeCrawlJointTargetRateAction,
        SnakeCrawlJointTargetRateActionCfg,
    )

    def initialize(self, cfg, env):
        self._env = env
        self._target_names = [f'joint{i}' for i in range(1, 7)]
        self._target_ids = list(range(6))
        self._raw_actions = torch.zeros(6, 6)
        self._processed_actions = torch.zeros(6, 6)
        self._entity = env.scene['robot']

    monkeypatch.setattr(JointPositionAction, '__init__', initialize)
    data = omni_env.scene['robot'].data
    data.joint_pos_limits = torch.tensor([-math.pi, math.pi]).expand(6, 6, 2).clone()
    cfg = SnakeCrawlJointTargetRateActionCfg(
        entity_name='robot', actuator_names=('.*',), max_target_speed=2 * math.pi,
        target_limit=math.pi, biped_residual_scale=.375,
    )
    action = SnakeCrawlJointTargetRateAction(cfg, omni_env)
    omni_env.action_manager.get_term = lambda name: action
    teachers = GuidedTeacherEnsemble(omni_env)
    output = torch.linspace(-2., 2., 36).reshape(6, 6)
    converted = teachers._convert('omni', output)
    desired = teachers.defaults['omni'] + .25 * output
    desired = desired.clamp(min=data.default_joint_pos - .375, max=data.default_joint_pos + .375)
    action.commanded_target.copy_(data.default_joint_pos + .8)
    previous = action.commanded_target.clone()
    max_delta = cfg.max_target_speed * omni_env.step_dt
    action.process_actions(converted)
    expected = previous + (desired - previous).clamp(-max_delta, max_delta)
    torch.testing.assert_close(action.commanded_target, expected)
    for _ in range(20):
        action.process_actions(converted)
    torch.testing.assert_close(action.commanded_target, desired)


def test_default_skips_snake_checkpoint_and_all_snake_adapter_work(omni_env, monkeypatch):
    import zbot_rl_mjlab.teachers as legacy_teachers

    monkeypatch.setattr(legacy_teachers, 'SNAKE_CHECKPOINT', '/missing/unqualified_snake.pt')

    def forbidden(*args, **kwargs):
        raise AssertionError('default guidance must never construct snake observations')

    monkeypatch.setattr(GuidedTeacherEnsemble, '_canonical_snake_observation', forbidden)
    teachers = GuidedTeacherEnsemble(omni_env)
    assert teachers.teacher_names == ('omni', 'transform')
    assert not hasattr(teachers, '_guided_proxy')
    assert not hasattr(teachers, '_guided_episode')
    command = omni_env.command_manager.get_term('transform')
    command.current_task.fill_(2)
    command.standing_mode.zero_()
    action = omni_env.action_manager.get_term('joint_pos')
    action.commanded_target.zero_()
    omni_env.scene['robot'].data.joint_pos.zero_()
    omni_env.scene['robot'].data.body_link_pos_w[:, :, 2] = .055
    result = teachers.query()
    assert result.counts == {'omni': 0, 'transform': 0}
    assert result.teacher_ids is not None
    assert result.teacher_ids.tolist() == [-1] * 6
    assert result.weights.eq(0).all() and result.actions.eq(0).all()
    with pytest.raises(ValueError, match='未启用'):
        GuidedTeacherEnsemble(omni_env, checkpoints={'snake': '/missing/unqualified_snake.pt'})
    transform_only = GuidedTeacherEnsemble(omni_env, include_omni=False)
    assert transform_only.teacher_names == ('transform',)
    assert transform_only.query().weights.eq(0).all()


def test_omni_33_columns_match_native_archive_terms_with_site_commands(omni_env):
    from types import SimpleNamespace

    from zbot_rl_mjlab.guided_teachers import _SceneView, _View

    teachers = GuidedTeacherEnsemble(omni_env)
    cfg = teachers.configs['omni']
    assert 'yaw_error' not in cfg.observations['actor'].terms
    assert 'target_speed' not in cfg.observations['actor'].terms
    command = omni_env.command_manager.get_term('transform')
    command.velocity_command[:] = torch.tensor([.12, .04, -.3])
    command.gait_frequency[:] = torch.linspace(.3, .9, 6)
    omni_env.episode_length_buf[:] = torch.arange(6) * 13 + 7
    action = omni_env.action_manager.get_term('joint_pos')
    previous_position_action = torch.linspace(-.7, .7, 36).reshape(6, 6)
    action.commanded_target.copy_(teachers.defaults['omni'] + .25 * previous_position_action)
    robot = omni_env.scene['robot']
    native_data = _View(robot.data,
                        default_joint_pos=teachers.defaults['omni'].expand(6, 6),
                        default_joint_vel=torch.zeros(6, 6))
    native_scene = _SceneView(omni_env.scene, _View(robot, data=native_data))
    # At nominal +90X roll, native walking forward is -worldX, whereas the
    # requested new site-frame forward is +worldX. Native commands must flip XY.
    native_velocity = torch.tensor([-.12, -.04, -.3]).expand(6, 3)
    proxy = _View(omni_env, cfg=cfg, scene=native_scene,
                  _zbot_step_frequency=command.gait_frequency.clone(),
                  action_manager=SimpleNamespace(action=previous_position_action),
                  command_manager=SimpleNamespace(get_command=lambda name: native_velocity))
    pieces = []
    for term in cfg.observations['actor'].terms.values():
        value = term.func(proxy, **term.params)
        if term.clip is not None:
            value = value.clamp(*term.clip)
        if term.scale is not None:
            value = value * term.scale
        pieces.append(value)
    expected = torch.cat(pieces, -1)
    actual = teachers.observation('omni')
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    command.initial_heading.mul_(-1)
    torch.testing.assert_close(teachers.observation('omni'), actual)


def test_ascent_recovery_labels_straighten_asymmetry_and_only_active_ascent(omni_env):
    from zbot_rl_mjlab.teachers import position_to_rate

    teachers = GuidedTeacherEnsemble(omni_env)
    command = omni_env.command_manager.get_term('transform')
    command.current_task[:] = torch.tensor([3, 3, 0, 1, 2, 3])
    command.transition_mask[:] = torch.tensor([True, False, False, True, False, True])
    command.standing_mode[:] = torch.tensor([True, True, True, False, False, True])
    action = omni_env.action_manager.get_term('joint_pos')
    # This pose is rejected by the archive's mirror coverage; recovery must still
    # teach the explicit corrective rule rather than supervise a bending action.
    action.commanded_target[0] = torch.tensor([2., .04, -.08, .1, -.03, 0.])
    omni_env.scene['robot'].data.joint_pos[0] = torch.tensor([2., 0., 0., 0., 0., 0.])
    action.raw_action[5, 0] = torch.nan
    command.straightening_mask = torch.zeros(6, dtype=torch.bool)
    before = teachers.query()
    assert before.teacher_ids is not None
    assert before.teacher_ids[0] == -1
    command.straightening_mask.fill_(True)
    after = teachers.query()
    assert after.teacher_ids is not None
    expected = position_to_rate(torch.zeros_like(action.commanded_target),
                                action.commanded_target, action._target_limits(),
                                omni_env.step_dt, action.rate_cfg.max_target_speed)
    torch.testing.assert_close(after.actions[0], expected[0])
    assert after.weights[0] == 1 and after.teacher_ids[0] == teachers.teacher_names.index('transform')
    assert after.counts['transform'] == before.counts['transform'] + 1
    # Inactive ascent, walking, descent, snake, and invalid physics are unchanged.
    torch.testing.assert_close(after.actions[1:], before.actions[1:])
    torch.testing.assert_close(after.weights[1:], before.weights[1:])
    torch.testing.assert_close(after.teacher_ids[1:], before.teacher_ids[1:])
    assert after.weights[5] == 0
    assert torch.isfinite(after.actions).all()
    for model in teachers.models.values():
        assert not model.training and all(not p.requires_grad for p in model.parameters())
