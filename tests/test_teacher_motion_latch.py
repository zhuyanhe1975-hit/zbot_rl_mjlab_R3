"""Brief contact loss cannot silently change the teacher's task or action units."""

from typing import Any, cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from test_teacher_biped_actions import hybrid
from test_transformation_action_bounds import (
    make_action as make_action,  # noqa: PLC0414
)
from test_universal_task import state_env

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.commands import UniversalCommandCfg
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg
from zbot_rl_mjlab.tasks.universal.teacher_rewards import biped_fallen


def motion_env(mode='stand'):
    env, _ = state_env()
    env._omni_initial_standing = torch.full((2,), mode == 'stand')
    term = PrivilegedOmniCommandCfg(mode=mode, training=True, resampling_time_range=(6., 12.)).build(env)
    env.command_manager.get_term = lambda _: term
    data = env.scene['robot'].data
    data.joint_pos[:] = data.joint_pos.new_tensor(env.cfg.standing_joint_pos)
    data.body_link_pos_w[:, 6, 2] = .264 if mode == 'stand' else .055
    if mode == 'snake':
        data.body_com_pos_w[..., 2] = .055
        data.joint_pos.zero_()
    term.reset(torch.arange(2))
    term.requested_velocity[:] = torch.tensor([.15, 0., 0.])
    term.velocity_command.copy_(term.requested_velocity)
    env.common_step_counter = 10
    return env, term


def test_single_missed_contact_preserves_walking_speed_and_residual_actions(make_action):
    env, term = motion_env()
    data = env.scene['robot'].data
    data.joint_pos[:, 0] += .2  # Legal walking offset, outside transform completion tolerance.
    env.scene['feet_ground_contact'].data.found.zero_()
    assert not mdp.configuration_ready(env, relaxed=True).any()
    term._update_command(None)
    env.scene['feet_ground_contact'].data.found.fill_(True)
    for _ in range(250):
        env.common_step_counter += 1
        term._update_command(None)
    assert term.phase.eq(0).all()
    assert not term.transition_mask.any()
    torch.testing.assert_close(term.velocity_command, term.requested_velocity)
    action, _ = hybrid(make_action)
    cast(Any, action._env.command_manager).get_term = lambda _: term
    for _ in range(20):
        action.process_actions(torch.ones(2, 6))
    # Residual position saturates at nominal+.25 instead of integrating to pi.
    torch.testing.assert_close(action.commanded_target, torch.full((2, 6), .75))
    data.body_link_pos_w[:, 6, 2] = .1
    assert biped_fallen(env).all()  # Real falls are still recycled.


def test_snake_contact_loss_keeps_requested_motion():
    env, term = motion_env('snake')
    env.scene['feet_ground_contact'].data.found.zero_()
    env.scene['transform_ground_contact'].data.force_history.zero_()
    assert not mdp.configuration_ready(env, relaxed=True).any()
    for _ in range(250):
        env.common_step_counter += 1
        term._update_command(None)
    assert term.phase.eq(2).all()
    assert not term.transition_mask.any()
    torch.testing.assert_close(term.velocity_command, term.requested_velocity)


def test_explicit_selected_mode_change_still_requires_transform_hold(monkeypatch):
    env, term = motion_env()
    term.set_target('snake', torch.tensor([0]))
    assert term.transition_mask.tolist() == [True, False]
    assert term.manual_control.tolist() == [True, False]
    monkeypatch.setattr(mdp, 'configuration_ready', lambda env, relaxed=False: torch.ones(2, dtype=torch.bool))
    for _ in range(90):
        env.common_step_counter += 1
        term._update_command(None)
    assert term.phase.tolist() == [1, 0]
    for _ in range(15):
        env.common_step_counter += 1
        term._update_command(None)
    assert term.phase.tolist() == [2, 0]
    assert not term.transition_mask.any()


def test_legacy_universal_training_keeps_automatic_recovery(monkeypatch):
    env, _ = state_env()
    term = UniversalCommandCfg(training=True, resampling_time_range=(6., 12.)).build(cast(ManagerBasedRlEnv, env))
    env.command_manager.get_term = lambda _: term
    term.transition_mask[:] = False
    monkeypatch.setattr(mdp, 'configuration_ready', lambda env, relaxed=False: torch.zeros(2, dtype=torch.bool))
    term._update_command(None)
    assert term.phase.eq(3).all()
    assert term.transition_mask.all()
