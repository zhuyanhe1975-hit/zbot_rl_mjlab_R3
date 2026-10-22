"""Native archived site-snake guidance, frame conversion and state isolation."""
import math
from types import SimpleNamespace
from typing import cast

import pytest
import test_universal_snake_teacher as snake_fixtures
import test_universal_teachers as teacher_fixtures
import torch

from zbot_rl_mjlab.guided_teachers import GuidedTeacherEnsemble
from zbot_rl_mjlab.tasks.universal.privileged import privileged_state
from zbot_rl_mjlab.teachers import SNAKE_SITE_CHECKPOINT

env = teacher_fixtures.env
snake_env = snake_fixtures.snake_env


@pytest.fixture
def site_env(snake_env):
    data = snake_env.scene['robot'].data
    data.body_com_pos_w = data.body_link_pos_w.clone()
    data.site_lin_vel_w = torch.full((6, 2, 3), .1)
    data.site_pos_w[..., 2] = .0555
    snake_env.scene['transform_ground_contact'] = SimpleNamespace(
        data=SimpleNamespace(force_history=torch.zeros(6, 1, 1, 3)))
    command = snake_env.command_manager.get_term('transform')
    command.current_task = torch.full((6,), 2, dtype=torch.long)
    command.gait_frequency = torch.full((6,), .7)
    command.current_morphology = torch.full((6,), -1, dtype=torch.long)
    command.requested_velocity[:] = torch.tensor([
        [0., 0., 0.], [-.08, 0., 0.], [0., .08, 0.], [0., -.08, 0.],
        [0., 0., -.3], [.08, -.08, .3],
    ])
    command.velocity_command.copy_(command.requested_velocity)
    return snake_env


@pytest.fixture
def site_teachers(site_env):
    return GuidedTeacherEnsemble(site_env, include_omni=False, include_site_snake=True)


def test_three_teachers_and_native_beta_archive_are_frozen(site_env, site_teachers):
    assert GuidedTeacherEnsemble(site_env, include_site_snake=True).teacher_names == (
        'omni', 'transform', 'snake_site')
    model = site_teachers.models['snake_site']
    saved = torch.load(SNAKE_SITE_CHECKPOINT, map_location='cpu', weights_only=False)
    assert type(model.distribution).__name__ == 'BetaDistribution'
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved['actor_state_dict'][name])
    assert site_teachers.query().counts['snake_site'] == 6
    with pytest.raises(ValueError, match='both'):
        GuidedTeacherEnsemble(site_env, include_snake=True, include_site_snake=True)


def test_native_56d_observation_matches_archive_columns_and_has_no_side_effects(site_env, site_teachers):
    command = site_env.command_manager.get_term('transform')
    action = site_env.action_manager.get_term('joint_pos')
    action.rate_cfg.max_target_speed = math.pi
    site_env.episode_length_buf.fill_(23)
    # Learner task metadata must never leak into the fixed snake teacher state.
    command.phase.fill_(3)
    command.current_morphology.fill_(0)
    command.current_task.fill_(1)
    before = {k: v.clone() for k, v in vars(command).items() if isinstance(v, torch.Tensor)}
    env_keys = set(vars(site_env))
    actual = site_teachers._site_snake_observation()
    proxy = site_teachers._site_proxy
    torch.testing.assert_close(actual, privileged_state(proxy))
    assert actual.shape == (6, 56)
    torch.testing.assert_close(actual[:, 18:24], action.raw_action / 2)
    torch.testing.assert_close(actual[:, 38], torch.full((6,), -1.))
    torch.testing.assert_close(actual[:, 39], torch.full((6,), 2.))
    torch.testing.assert_close(actual[:, 40:43], command.velocity_command)
    phase = 23 * site_env.step_dt * .7 * 2 * math.pi
    torch.testing.assert_close(actual[:, 43:46], torch.tensor([math.sin(phase), math.cos(phase), .7]).expand(6, 3))
    torch.testing.assert_close(actual[:, 50:53], torch.tensor([.1, .1, .15]).expand(6, 3))
    torch.testing.assert_close(actual[:, 53:56], command.requested_velocity)
    assert set(vars(site_env)) == env_keys
    for key, value in before.items():
        torch.testing.assert_close(getattr(command, key), value)
    assert not hasattr(site_env, '_omni_motion_site_index')


def test_command_vectors_are_rotated_from_learner_into_current_site_frame(site_env, site_teachers):
    command = site_env.command_manager.get_term('transform')
    command.heading[:] = torch.tensor([0., 1.])
    command.velocity_command[:] = torch.tensor([.08, -.04, .3])
    command.requested_velocity.copy_(command.velocity_command)
    actual = site_teachers._site_snake_observation()
    torch.testing.assert_close(actual[:, 40:43], torch.tensor([.04, .08, .3]).expand(6, 3))
    torch.testing.assert_close(actual[:, 53:56], actual[:, 40:43])


def test_transition_invalid_pose_and_nonfinite_physics_do_not_produce_labels(site_env, site_teachers):
    command = site_env.command_manager.get_term('transform')
    command.transition_mask[0] = True
    command.velocity_command[1, 0] = .21
    command.requested_velocity[2, 1] = torch.nan
    command.heading[3] = 0
    site_env.scene['robot'].data.site_lin_vel_w[4, 0, 0] = torch.inf
    site_env.action_manager.get_term('joint_pos').commanded_target[5, 0] = math.pi / 3 + .01
    result = site_teachers.query()
    assert result.counts['snake_site'] == 0
    assert result.teacher_ids.tolist() == [-1] * 6
    assert torch.isfinite(result.actions).all()


def test_native_rate_conversion_respects_both_control_limits(site_env, site_teachers):
    action = site_env.action_manager.get_term('joint_pos')
    action.rate_cfg.max_target_speed = math.pi
    torch.testing.assert_close(site_teachers._convert('snake_site', torch.full((6, 6), .25)), torch.full((6, 6), .5))
    action.commanded_target.fill_(math.pi / 3 - .01)
    expected = torch.full((6, 6), .01 / (site_env.step_dt * math.pi))
    torch.testing.assert_close(site_teachers._convert('snake_site', torch.ones(6, 6)), expected)
    action._target_limits = lambda: torch.tensor([-1., 1.]).expand(6, 6, 2)
    action.commanded_target.fill_(.99)
    torch.testing.assert_close(site_teachers._convert('snake_site', torch.ones(6, 6)), expected)


def test_incompatible_metadata_is_rejected(site_env, tmp_path):
    saved = torch.load(SNAKE_SITE_CHECKPOINT, map_location='cpu', weights_only=False)
    saved['infos']['privileged_omni_teacher']['actor_obs_dim'] = 61
    checkpoint = tmp_path / 'bad.pt'
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match='compatible'):
        GuidedTeacherEnsemble(site_env, include_site_snake=True, checkpoints={'snake_site': str(checkpoint)})


def test_rotated_site_frame_preserves_native_velocity_and_world_angular_channels(site_env, site_teachers):
    from test_guided_teachers import rotate_world

    site_teachers.query()
    observation = site_teachers._site_snake_observation().clone()
    yaw = .718
    rotate_world(site_env, yaw)
    value = site_env.scene['robot'].data.site_lin_vel_w
    x, y = value[..., 0].clone(), value[..., 1].clone()
    value[..., 0] = math.cos(yaw) * x - math.sin(yaw) * y
    value[..., 1] = math.sin(yaw) * x + math.cos(yaw) * y
    rotated = GuidedTeacherEnsemble(site_env, include_omni=False, include_site_snake=True)
    actual = rotated.query()
    assert actual.counts['snake_site'] == 6
    native = rotated._site_snake_observation()
    # Archive base angular velocity is world-frame; preserve that recipe too.
    torch.testing.assert_close(native[:, 31:34], site_env.scene['robot'].data.body_link_ang_vel_w[:, 0])
    for start, end in ((24, 27), (40, 46), (50, 56)):
        torch.testing.assert_close(native[:, start:end], observation[:, start:end], atol=1e-6, rtol=1e-5)


def test_partial_episode_reset_and_task_entry_reset_only_native_heading(site_env, site_teachers):
    site_teachers.query()
    native = site_teachers._site_command
    native.initial_heading[:] = torch.tensor([0., 1.])
    site_env.episode_length_buf.fill_(10)
    site_env.common_step_counter = 10
    site_teachers.query()
    site_env.episode_length_buf[2] = 0
    site_env.common_step_counter = 11
    command = site_env.command_manager.get_term('transform')
    command.current_task[3] = 0
    site_teachers.query()
    torch.testing.assert_close(native.initial_heading[[2, 3]], torch.tensor([1., 0.]).expand(2, 2))
    torch.testing.assert_close(native.initial_heading[[0, 1, 4, 5]], torch.tensor([0., 1.]).expand(4, 2))
    assert not hasattr(command, 'initial_heading')


def test_teacher_labels_require_the_visible_archive_gait_clock(site_env, site_teachers):
    command = site_env.command_manager.get_term('transform')
    command.gait_frequency[1] = .5
    command.gait_frequency[2] = torch.nan
    result = site_teachers.query()
    assert result.teacher_ids.tolist() == [1, -1, -1, 1, 1, 1]


def test_full_observation_parity_with_independent_archived_command(site_env, site_teachers):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from zbot_rl_mjlab.archived_site_snake import (
        ArchivedSiteSnakeCommand,
        ArchivedSiteSnakeCommandCfg,
    )

    cfg = load_env_cfg('Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher', play=True)
    learner_command = site_env.command_manager.get_term('transform')
    action = site_env.action_manager.get_term('joint_pos')
    action.rate_cfg.max_target_speed = math.pi
    site_env.episode_length_buf.fill_(37)
    native_env = SimpleNamespace(**{**vars(site_env), 'cfg': cfg})
    command_cfg = cast(ArchivedSiteSnakeCommandCfg, cfg.commands['transform'])
    native_command = cast(ArchivedSiteSnakeCommand, command_cfg.build(cast(ManagerBasedRlEnv, native_env)))
    native_command._command[:, 0] = -1.
    native_command.task_fsm.reset(torch.arange(6), torch.zeros(6, dtype=torch.bool), torch.zeros(6, dtype=torch.bool))
    native_command._publish_task_state()
    native_command.velocity_command.copy_(learner_command.velocity_command)
    native_command.requested_velocity.copy_(learner_command.requested_velocity)
    native_command.gait_frequency.fill_(.7)
    native_env.command_manager = SimpleNamespace(get_term=lambda name: native_command)
    native_env.action_manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(
        commanded_target=action.commanded_target, raw_action=action.raw_action / 2))
    expected = privileged_state(native_env)
    torch.testing.assert_close(site_teachers._site_snake_observation(), expected)
