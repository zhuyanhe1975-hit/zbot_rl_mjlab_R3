"""Motion starts and transformation completion obey one consistent teacher recipe."""

from types import SimpleNamespace

import torch
from test_universal_task import state_env

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.privileged import PrivilegedOmniCommandCfg


def test_canonical_matched_reset_enters_motion_and_opposite_pose_still_transforms():
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0,
                          _omni_initial_standing=torch.tensor([True, False]))
    term = PrivilegedOmniCommandCfg(mode='stand', training=True, resampling_time_range=(6., 12.)).build(env)
    term.reset(torch.arange(2))
    assert term.phase.tolist() == [0, 3]
    assert term.transition_mask.tolist() == [False, True]
    assert term.standing_achieved.tolist() == [True, False]
    assert ((term.gait_frequency >= .25) & (term.gait_frequency <= 1.)).all()
    before = term.gait_frequency[1].clone()
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.gait_frequency[1], before)


def test_snake_reset_uses_archived_point_seven_hz_and_preserves_ascent_hold():
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=.02, common_step_counter=0,
                          _omni_initial_standing=torch.tensor([True, False]))
    cfg = PrivilegedOmniCommandCfg(mode='snake', training=True, resampling_time_range=(6., 12.))
    term = cfg.build(env)
    term.reset(torch.arange(2))
    assert term.phase.tolist() == [1, 2]
    assert term.transition_mask.tolist() == [True, False]
    torch.testing.assert_close(term.gait_frequency, torch.full((2,), .7))
    assert cfg.hold_s == 2.


def test_transformation_completion_cannot_fall_between_ready_and_failure_height():
    env, term = state_env()
    term.cfg = PrivilegedOmniCommandCfg(resampling_time_range=(6., 12.))
    term.command[:, 0] = 1
    term.transition_mask[:] = True
    data = env.scene['robot'].data
    data.joint_pos[:] = data.joint_pos.new_tensor(env.cfg.standing_joint_pos)
    data.body_link_pos_w[:, 6, 2] = torch.tensor([.21, .264])
    assert mdp.configuration_ready(env).tolist() == [False, True]
    data.joint_pos[1, 0] += .3
    assert not mdp.configuration_ready(env).any()
    term.transition_mask[1] = False
    assert mdp.configuration_ready(env)[1]  # Normal walking is not locked to the pose.


def test_reset_contact_warmup_does_not_turn_canonical_walking_into_transformation():
    env, _ = state_env()
    env._omni_initial_standing = torch.ones(2, dtype=torch.bool)
    term = PrivilegedOmniCommandCfg(mode='stand', resampling_time_range=(6., 12.)).build(env)
    env.command_manager.get_term = lambda _: term
    data = env.scene['robot'].data
    data.joint_pos[:] = data.joint_pos.new_tensor(env.cfg.standing_joint_pos)
    data.body_link_pos_w[:, 6, 2] = .264
    env.scene['feet_ground_contact'].data.found.zero_()
    env.scene['transform_ground_contact'].data.force_history.zero_()
    term.reset(torch.arange(2))
    env.common_step_counter += 1
    term._update_command(None)
    assert term.phase.eq(0).all()
    # Contact loss cannot silently turn locomotion into a transformation.
    env.common_step_counter += 6
    term._update_command(None)
    assert not term.transition_mask.any()
    term.set_target('snake')
    assert term.canonical_motion_start.eq(-1).all()
