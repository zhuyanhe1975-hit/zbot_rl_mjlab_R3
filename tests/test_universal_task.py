"""综合任务的独立动作、模式门控与连续指令契约。"""
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from zbot_rl_mjlab.tasks.transformation.actions import JointTargetRateActionCfg
from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.commands import UniversalCommandCfg
from zbot_rl_mjlab.tasks.universal.env_cfg import universal_env_cfg


def command_env():
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.02, common_step_counter=0)
    term = UniversalCommandCfg(resampling_time_range=(5., 8.), training=False).build(cast(ManagerBasedRlEnv, env))
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    return env, term


def test_six_independent_targets_and_fixed_observation_contract():
    cfg = universal_env_cfg()
    action = cast(JointTargetRateActionCfg, cfg.actions["joint_pos"])
    assert action.max_target_speed > 6
    assert action.target_limit is not None and action.target_limit > 3
    assert not action.joint_target_bounds
    assert mdp.UNIVERSAL_ACTOR_DIM == 50
    assert mdp.UNIVERSAL_CRITIC_DIM == 53
    assert cfg.scene.num_envs == 8192
    assert cfg.episode_length_s == 45
    assert "joint_zero" not in cfg.rewards


def test_keyboard_change_does_not_reset_and_stop_preserves_mode():
    env, term = command_env()
    term.set_target("snake")
    term.set_velocity(.2, -.1, .4)
    assert term.command[:, 0].eq(-1).all()
    assert term.manual_control.all() and term.time_left.isinf().all()
    term.stop()
    assert term.command[:, 0].eq(-1).all()
    assert term.requested_velocity.eq(0).all()
    assert env.common_step_counter == 0


def test_transition_gates_then_restores_pending_velocity(monkeypatch):
    env, term = command_env()
    term.set_target("snake")
    term.set_velocity(.2, 0., 0.)
    monkeypatch.setattr(mdp, "configuration_ready", lambda env, relaxed=False: torch.zeros(2, dtype=torch.bool))
    term._update_command(None)
    assert term.transition_mask.all()
    assert term.velocity_command.eq(0).all()
    monkeypatch.setattr(mdp, "configuration_ready", lambda env, relaxed=False: torch.ones(2, dtype=torch.bool))
    for _ in range(30):
        env.common_step_counter += 1
        term._update_command(None)
    assert term.phase.eq(2).all()
    assert not term.transition_mask.any()
    assert (term.velocity_command[:, 0] > 0).all()
    assert term.requested_velocity[:, 0].eq(.2).all()


def test_stationary_cannot_score_as_commanded_motion():
    requested = torch.tensor([[.1, 0., 0.], [0., 0., 0.]])
    moving = mdp.motion_tracking_score(torch.zeros_like(requested), requested)
    assert moving.eq(0).all()
    assert mdp.motion_tracking_score(requested, requested)[0] > .99


def test_manual_sampling_is_disabled_and_reset_restores_training_sampling():
    _env, term = command_env()
    term.set_velocity(.1, 0., 0.)
    before = term.command.clone()
    term._resample_command(torch.arange(2))
    torch.testing.assert_close(term.command, before)
    term.reset(torch.arange(2))
    assert term.command[:, 0].eq(1).all()
    assert term.requested_velocity.eq(0).all()
    assert term.time_left.isinf().all()


def state_env():
    from test_snake_to_biped import _standing_env

    env = _standing_env()
    env.cfg = universal_env_cfg()
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.scene["base_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(2, 1)))
    term = cast(UniversalCommandCfg, env.cfg.commands["transform"]).build(cast(ManagerBasedRlEnv, env))
    env.command_manager = SimpleNamespace(get_term=lambda name: term)
    env.scene["transform_ground_contact"] = SimpleNamespace(data=SimpleNamespace(force_history=torch.zeros(2, 4, 12, 3)))
    env.scene["transform_ground_contact"].data.force_history[..., 2] = 20
    data = env.scene["robot"].data
    data.actuator_force = torch.zeros(2, 6)
    action = SimpleNamespace(commanded_target=data.joint_pos.clone(), raw_action=torch.zeros(2, 6))
    env.action_manager = SimpleNamespace(get_term=lambda name: action, action=torch.zeros(2, 6), prev_action=torch.zeros(2, 6))
    return env, term


def test_actual_state_dimensions_finite_and_all_targets_observed():
    env, term = state_env()
    action = env.action_manager.get_term("joint_pos")
    action.commanded_target[:] = torch.arange(6)
    actor = mdp.actor_state(env)
    assert actor.shape == (2, mdp.UNIVERSAL_ACTOR_DIM)
    assert mdp.critic_state(env).shape == (2, mdp.UNIVERSAL_CRITIC_DIM)
    torch.testing.assert_close(actor[:, 12:18], action.commanded_target)
    assert torch.isfinite(actor).all()
    assert term.heading.norm(dim=-1).allclose(torch.ones(2))


def test_low_snake_is_legal_and_no_zero_pose_reward_during_crawling():
    env, term = state_env()
    term.set_target("snake")
    data = env.scene["robot"].data
    data.body_link_pos_w[..., 2] = .055
    data.body_com_pos_w[..., 2] = .055
    term.phase[:] = 2
    term.transition_mask[:] = False
    assert mdp.configuration_ready(env).all()
    assert not mdp.unsafe(env).any()
    assert mdp.transition_progress_reward(env).eq(0).all()
    before = mdp.posture_reward(env)
    data.joint_pos[:] = torch.tensor([1., -.5, .7, -.2, .3, -.9])
    torch.testing.assert_close(before, mdp.posture_reward(env))
    term.phase[:] = 0
    term.standing_achieved[:] = True
    term._command[:, 0] = 1
    assert mdp.unsafe(env).all()


def test_rewards_gate_transition_motion_and_zero_speed_separately():
    env, term = state_env()
    term.set_velocity(.1, 0., 0.)
    term.velocity_command[:, 0] = .1
    assert mdp.velocity_tracking(env).eq(0).all()
    assert mdp.standstill_reward(env).eq(0).all()
    term.stop()
    term.transition_mask[:] = False
    assert mdp.standstill_reward(env).eq(1).all()
    assert mdp.velocity_tracking(env).eq(0).all()


def test_training_sampling_curriculum_and_modes():
    env, term = command_env()
    term.cfg.training = True
    term._resample_command(torch.arange(2))
    assert set(term.command[:, 0].tolist()) <= {-1., 1.}
    assert (term.requested_velocity.abs() <= torch.tensor(term.cfg.velocity_limits) * .2 + 1e-6).all()
    env.common_step_counter = 1000000
    term._resample_command(torch.arange(2))
    assert (term.requested_velocity.abs() <= torch.tensor(term.cfg.velocity_limits) * .2 + 1e-6).all()
    assert not term.manual_control.any()


def test_gait_and_initial_heading_are_visible_and_reset_scoped():
    env, term = state_env()
    env.episode_length_buf[:] = 25
    gait = mdp.gait_observation(env)
    torch.testing.assert_close(gait[:, 2], torch.full((2,), .5))
    torch.testing.assert_close(gait[:, 0], torch.ones(2))
    mdp.heading(env)
    assert term.heading_initialized.all()
    torch.testing.assert_close(term.initial_heading, term.heading)
    torch.testing.assert_close(mdp.heading_error(env), torch.zeros(2))
    original = term.initial_heading[1].clone()
    term.reset(torch.tensor([0]))
    assert not term.heading_initialized[0] and term.heading_initialized[1]
    torch.testing.assert_close(term.initial_heading[1], original)
    assert mdp.actor_state(env).shape[-1] == 50


def test_base_contact_is_visible_but_not_standing_support():
    env, term = state_env()
    env.scene["feet_ground_contact"].data.found.zero_()
    env.scene["base_ground_contact"].data.found.fill_(1)
    torch.testing.assert_close(mdp.contacts(env), torch.tensor([[0., 0., 1.], [0., 0., 1.]]))
    assert term.standing_mode.all()
    assert not mdp.configuration_ready(env).any()


def test_achieved_standing_latch_detects_fall_after_phase_changes():
    env, term = state_env()
    term.standing_achieved[:] = True
    term.phase[:] = 3
    term.transition_mask[:] = True
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = .08
    assert mdp.unsafe(env).all()
    term.standing_achieved[:] = False
    assert not mdp.unsafe(env).any()
    term.standing_achieved[:] = True
    term.set_target("snake")
    assert not term.standing_achieved.any()
    assert not mdp.unsafe(env).any()


def test_manual_play_disables_timeout_and_physical_resets_but_not_nonfinite():
    env, term = state_env()
    term.cfg.training = False
    env.max_episode_length = 2250
    env.episode_length_buf[:] = 2250
    assert mdp.play_time_out(env).all()
    term.set_velocity(.1, 0., 0.)
    assert not mdp.play_time_out(env).any()
    term.standing_achieved[:] = True
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = .08
    env.scene["robot"].data.body_link_pos_w[:, 6, 0] = 100.
    assert not mdp.unsafe(env).any()
    env.scene["robot"].data.joint_pos[0, 0] = float("nan")
    assert mdp.unsafe(env).tolist() == [True, False]


def test_tracking_uses_com_translation_instead_of_swinging_base():
    env, term = state_env()
    term.transition_mask[:] = False
    term.velocity_command[:, 0] = .1
    forward = mdp.heading(env)
    env.scene["robot"].data.body_link_lin_vel_w[:, 6, :2] = forward * .1
    assert mdp.velocity(env)[:, 0].allclose(torch.full((2,), .1))
    assert mdp.velocity_tracking(env).eq(0).all()
    env.scene["robot"].data.body_com_lin_vel_w[..., :2] = forward[:, None, :] * .1
    assert mdp.velocity_tracking(env).allclose(torch.ones(2))


def test_failure_has_fixed_per_event_integrated_penalty():
    env, term = state_env()
    term.standing_achieved[:] = True
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = .08
    cost = mdp.failure_cost(env)
    torch.testing.assert_close(cost * env.step_dt, torch.ones(2))
    assert env.cfg.rewards["failure"].weight == -5


def test_training_resampling_restores_mode_quota_without_touching_manual():
    env = SimpleNamespace(num_envs=8, device="cpu", step_dt=.02, common_step_counter=0)
    term = UniversalCommandCfg(resampling_time_range=(5., 8.)).build(cast(ManagerBasedRlEnv, env))
    ids = torch.arange(8)
    term._resample_command(ids)
    assert term.standing_mode.sum() == 4
    term.transition_mask[:] = False
    repeat = torch.tensor([0, 1, 2])
    for _ in range(10):
        term.transition_mask[:] = False
        term._resample_command(repeat)
        assert term.standing_mode.sum() == 4
    term.set_target("snake", torch.tensor([0]))
    term.set_velocity(.12, -.08, .3, torch.tensor([0]))
    before = term.command[0].clone()
    requested = term.requested_velocity[0].clone()
    term._resample_command(ids)
    torch.testing.assert_close(term.command[0], before)
    torch.testing.assert_close(term.requested_velocity[0], requested)
    assert term.manual_control[0]
    assert term.standing_mode.sum() == 4


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("mode,value", [("stand", 1.), ("snake", -1.)])
def test_fixed_mode_config_is_consumed(training, mode, value):
    env = SimpleNamespace(num_envs=4, device="cpu", step_dt=.02, common_step_counter=0)
    cfg = UniversalCommandCfg(resampling_time_range=(5., 8.), training=training, mode=mode)
    term = cfg.build(cast(ManagerBasedRlEnv, env))
    term.reset(torch.arange(4))
    assert term.command[:, 0].eq(value).all()
    if not training:
        assert term.requested_velocity.eq(0).all()
        assert term.time_left.isinf().all()


@pytest.mark.parametrize("options", [
    {"mode": "invalid"}, {"auto_switch_on_success": True},
    {"hold_s": float("nan")}, {"smoothing_s": float("inf")},
    {"transition_timeout_s": 0}, {"transition_timeout_s": float("nan")},
])
def test_invalid_command_options_fail_early(options):
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02, common_step_counter=0)
    with pytest.raises(ValueError):
        UniversalCommandCfg(resampling_time_range=(5., 8.), **options).build(cast(ManagerBasedRlEnv, env))


def test_mode_motion_metrics_exclude_inactive_and_empty_samples():
    env, term = state_env()
    term.phase[:] = torch.tensor([0, 2])
    term.requested_velocity[:, 0] = .1
    term.velocity_command[:, 0] = .1
    data = env.scene["robot"].data
    forward = mdp.heading(env)
    data.body_com_lin_vel_w[0, :, :2] = forward[0] * .05
    data.body_com_lin_vel_w[1, :, :2] = forward[1] * 100.
    expected = torch.tensor((.05 / term.cfg.velocity_limits[0]) / (3 ** .5))
    torch.testing.assert_close(mdp.mode_motion_error(env, 0).mean(), expected)
    assert mdp.phase_fraction(env, 0).mean() == .5
    term.phase[:] = 3
    empty = mdp.mode_motion_error(env, 0)
    assert torch.isfinite(empty).all() and empty.eq(0).all()


def test_tracking_axes_are_equal_and_have_no_clipped_plateau():
    requested = torch.tensor([[.05, 0., 0.], [0., .04, 0.], [0., 0., .16]])
    torch.testing.assert_close(mdp.motion_tracking_score(requested * .5, requested), torch.full((3,), mdp.motion_tracking_score(requested[:1] * .5, requested[:1]).item()))
    near = mdp.motion_tracking_score(requested * -1, requested)
    far = mdp.motion_tracking_score(requested * -2, requested)
    assert (near > far).all()


def test_vertical_velocity_is_observed():
    env, _term = state_env()
    before = mdp.actor_state(env)
    env.scene['robot'].data.body_link_lin_vel_w[:, 6, 2] = .2
    after = mdp.actor_state(env)
    assert not torch.equal(before, after)


def test_transition_has_deadline_but_manual_play_does_not():
    env, term = state_env()
    env.common_step_counter += int(term.cfg.transition_timeout_s / env.step_dt) + 1
    assert mdp.unsafe(env).all()
    term.cfg.training = False
    term.set_target('stand')
    assert not mdp.unsafe(env).any()


def test_pretraining_four_groups_have_teacher_covered_commands():
    env = SimpleNamespace(num_envs=8, device='cpu', step_dt=.02, common_step_counter=0)
    term = UniversalCommandCfg(resampling_time_range=(6., 12.)).build(cast(ManagerBasedRlEnv, env))
    term.set_pretraining(True)
    term.reset(torch.arange(8))
    assert term.standing_mode.tolist() == [True, True, False, True] * 2
    torch.testing.assert_close(term.requested_velocity[:4], torch.tensor([[.08, 0., 0.], [0., .05, 0.], [0., 0., 0.], [0., 0., 0.]]))


def test_potential_progress_has_no_phase_boundary_or_oscillation_profit(monkeypatch):
    env, term = state_env()
    potential = torch.tensor([.2, .2])
    monkeypatch.setattr(mdp, "posture_reward", lambda env: potential)
    assert mdp.transition_progress_reward(env).eq(0).all()
    env.common_step_counter += 1
    potential[:] = .8
    forward = mdp.transition_progress_reward(env).clone()
    assert (forward > 0).all()
    # 同步读奖励幂等；到位时姿态不变，奖励不会突然减少常数项。
    torch.testing.assert_close(mdp.transition_progress_reward(env), forward)
    term.transition_mask[:] = False
    env.common_step_counter += 1
    assert mdp.transition_progress_reward(env).eq(0).all()
    potential[:] = .2
    env.common_step_counter += 1
    backward = mdp.transition_progress_reward(env)
    torch.testing.assert_close(forward + backward, torch.zeros(2))


def test_curriculum_unlocks_from_tracking_instead_of_elapsed_time():
    env = SimpleNamespace(num_envs=256, device="cpu", step_dt=.02, common_step_counter=1000000)
    term = UniversalCommandCfg(resampling_time_range=(6., 12.)).build(cast(ManagerBasedRlEnv, env))
    ids = torch.arange(env.num_envs)
    term._resample_command(ids)
    assert (term.requested_velocity.count_nonzero(dim=-1) <= 1).all()
    assert (term.requested_velocity.abs() <= torch.tensor(term.cfg.velocity_limits) * .2 + 1e-6).all()
    term.tracking_ema[:] = .9
    term.transition_mask[:] = False
    term._resample_command(ids)
    assert (term.requested_velocity.count_nonzero(dim=-1) == 3).any()
    assert (term.requested_velocity.abs() > torch.tensor(term.cfg.velocity_limits) * .2).any()


def test_automatic_resampling_cannot_escape_unfinished_transition():
    env, term = command_env()
    term.cfg.training = True
    before = term.command.clone()
    env.common_step_counter += 350
    term._resample_command(torch.arange(2))
    torch.testing.assert_close(term.command, before)
    assert term.start_step.eq(0).all()


def test_combined_translation_and_yaw_receive_equal_credit():
    requested = torch.tensor([[.05, .04, .16]]).repeat(3, 1)
    actual = requested * torch.eye(3)
    torch.testing.assert_close(mdp.motion_tracking_score(actual, requested), torch.full((3,), 1 / 3))


def test_only_pretraining_repeats_ten_second_demonstrations():
    env, term = state_env()
    env.episode_length_buf[:] = torch.tensor([499, 500])
    assert not mdp.demonstration_time_out(env).any()
    term.set_pretraining(True)
    assert mdp.demonstration_time_out(env).tolist() == [False, True]
    term.set_velocity(.1, 0., 0.)
    assert not mdp.demonstration_time_out(env).any()
    term.manual_control[:] = False
    term.cfg.training = False
    assert not mdp.demonstration_time_out(env).any()
    cfg = universal_env_cfg()
    assert cfg.terminations['demonstration_time_out'].time_out
    assert cfg.episode_length_s == 45


def test_completed_transform_demonstration_restarts_before_long_static_tail():
    env, term = state_env()
    term.set_pretraining(True)
    term.transition_mask[:] = False
    env.episode_length_buf[:] = 100
    # 将代理环境的编号扩至四种示范，避免只覆盖前进/侧移。
    env.num_envs = 4
    env.episode_length_buf = torch.full((4,), 100)
    term.transition_mask = torch.tensor([False, False, False, True])
    term.manual_control = torch.zeros(4, dtype=torch.bool)
    assert mdp.demonstration_time_out(env).tolist() == [False, False, True, False]
    term.transition_mask[3] = False
    assert mdp.demonstration_time_out(env).tolist() == [False, False, True, True]


def test_teacher_forward_progress_outweighs_normal_sway():
    requested = torch.tensor([[.08, 0., 0.]])
    actual = torch.tensor([[.084, .066, .525]])
    assert mdp.motion_tracking_score(actual, requested).item() > .5
    assert mdp.motion_tracking_score(torch.zeros_like(actual), requested).item() == 0
    assert mdp.motion_tracking_score(torch.tensor([[0., .066, .525]]), requested).item() <= 0


def test_tiny_mixed_command_cannot_dominate_main_axis():
    requested = torch.tensor([[.08, .00001, .00001]])
    actual = torch.tensor([[.08, 0., 0.]])
    assert mdp.motion_tracking_score(actual, requested).item() > .99
