"""双向目标路由、观测命令及无重置切换。"""

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from zbot_rl_mjlab.experiments.bidirectional import SnakeBipedBidirectional
from zbot_rl_mjlab.rl import RobustSnakeToBipedRunner
from zbot_rl_mjlab.tasks.transformation import commanded, mdp, robust
from zbot_rl_mjlab.tasks.transformation import snake_to_biped as standing
from zbot_rl_mjlab.tasks.transformation.commanded import BidirectionalEnvCfg


def environment():
    from test_snake_to_biped import _standing_env

    env = _standing_env()
    env.cfg = SnakeBipedBidirectional().build_env()
    assert isinstance(env.cfg, BidirectionalEnvCfg)
    env._transform_initial_standing = torch.tensor([False, True])
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.scene["base_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=torch.ones(2, 1)))
    action = SimpleNamespace(commanded_target=torch.full((2, 6), 0.2))
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    term = env.cfg.commands["transform"].build(env)
    env.command_manager = SimpleNamespace(get_term=lambda name: term, get_command=lambda name: term.command)
    env.scene["robot"].data.body_link_ang_vel_w = torch.zeros_like(env.scene["robot"].data.body_link_lin_vel_w)
    return env, term


@pytest.mark.parametrize("play", [False, True])
def test_command_task_is_registered_and_preserves_archive_contract(play):
    cfg = load_env_cfg(commanded.TASK_ID, play=play)
    assert cfg.pretrained_checkpoint is None
    assert cfg.initial_roll_range_deg == (-180., 180.)
    assert not load_rl_cfg(commanded.TASK_ID).resume
    assert cfg.episode_length_s == 16
    assert cfg.auto_reset
    assert cfg.terminations["time_out"].time_out
    assert cfg.commands["transform"].mode == "source"
    assert cfg.commands["transform"].resampling_time_range == (1e9, 1e9)
    assert cfg.events["reset_transition"].func is commanded.reset_command_transition
    assert cfg.fall_retry_arm_height == cfg.early_contact_arm_height == 0.05
    assert cfg.fall_retry_drop_threshold == cfg.early_contact_drop_threshold == 0.01
    assert cfg.rewards["joint_standing"].weight == 12
    assert len(cfg.rewards) == 7
    assert "goal_motion" not in cfg.rewards
    assert load_runner_cls(commanded.TASK_ID) is RobustSnakeToBipedRunner


def test_initial_commands_match_source_pose_then_switch_without_teleport():
    env, term = environment()
    ids = torch.arange(2)
    data = env.scene["robot"].data
    original_q = data.joint_pos.clone()
    original_root = data.body_link_quat_w.clone()
    term.reset(ids)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([1., -1.]))
    mdp.state(env).hold_time[:] = 2
    robust.fall_retry_state(env).peak_base_height[:] = 0.3
    env.common_step_counter += 1
    term.set_target("snake", torch.tensor([0]))
    term.set_target("stand", torch.tensor([1]))
    torch.testing.assert_close(term.command[:, 0], torch.tensor([-1., 1.]))
    torch.testing.assert_close(data.joint_pos, original_q)
    torch.testing.assert_close(data.body_link_quat_w, original_root)
    assert mdp.state(env).hold_time.eq(0).all()
    assert robust.fall_retry_state(env).peak_base_height.isneginf().all()
    assert term.start_step.eq(env.common_step_counter).all()


def test_manual_command_is_scoped_and_holds_until_next_reset():
    env, term = environment()
    term.reset(torch.arange(2))
    mdp.state(env).hold_time[:] = 2
    term.set_target("snake", torch.tensor([0]))
    assert term.time_left[0].isinf()
    assert mdp.state(env).hold_time.tolist() == [0, 2]
    term.compute(50.)
    assert term.command[0, 0] == -1
    with pytest.raises(ValueError):
        term.set_target("walk")


def test_target_pose_height_and_sole_rewards_follow_command():
    env, term = environment()
    term._command[:, 0] = torch.tensor([1., -1.])
    data = env.scene["robot"].data
    data.joint_pos[1] = 0
    data.body_link_pos_w[1, 6, 2] = standing.SNAKE_ROOT_POS[2]
    torch.testing.assert_close(commanded.pose_reward(env), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(commanded.height_reward(env), torch.ones(2))
    torch.testing.assert_close(commanded.sole_reward(env), torch.tensor([1., 0.]))
    term._command *= -1
    assert (commanded.pose_reward(env) < 0).all()
    torch.testing.assert_close(commanded.height_reward(env), torch.zeros(2))


def test_descent_has_no_upward_velocity_reward_or_standing_fall_resets(monkeypatch):
    env, term = environment()
    term._command[:, 0] = torch.tensor([1., -1.])
    monkeypatch.setattr(robust, "initial_ascent_velocity_reward", lambda *args: torch.tensor([0.75, -4.]))
    robust.fall_retry_state(env).low_time[:] = env.cfg.fall_retry_hold_s
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = 0.05
    env.scene["robot"].data.body_link_lin_vel_w[:, 6, 2] = -0.05
    robust.fall_retry_state(env).peak_base_height[:] = 0.12
    torch.testing.assert_close(commanded.velocity_reward(env), torch.tensor([0.75, 0.]))
    assert commanded.fallen(env).tolist() == [True, False]
    assert commanded.early_contact(env).tolist() == [True, False]


def test_switch_clears_success_hold_and_requires_the_new_goal():
    env, term = environment()
    term._command.fill_(1)
    mdp.state(env).hold_time[:] = 3
    assert commanded.success(env).all()
    term.set_target("snake")
    assert not commanded.success(env).any()
    commanded.update_state(env)
    assert mdp.state(env).hold_time.eq(0).all()


def test_actor_and_critic_receive_command_and_keep_old_columns():
    from test_transformation_teacher import teacher_env

    env = teacher_env.__wrapped__()
    value = torch.tensor([[1.], [-1.]])
    env.command_manager = SimpleNamespace(get_command=lambda name: value)
    original_actor = commanded.teacher_actor_state(env).clone()
    original_critic = commanded.teacher_critic_state(env).clone()
    actor = commanded.actor_state(env)
    critic = commanded.critic_state(env)
    assert actor.shape == (2, 37) and critic.shape == (2, 45)
    torch.testing.assert_close(actor[:, :36], original_actor)
    torch.testing.assert_close(critic[:, :44], original_critic)
    torch.testing.assert_close(actor[:, -1:], value)
    torch.testing.assert_close(critic[:, -1:], value)


def test_stagnation_grace_restarts_at_command_change():
    env, term = environment()
    env.common_step_counter = 200
    term.set_target("snake")
    for step in range(200, 214):
        env.common_step_counter = step
        assert not commanded.abandoned_exploration(env).any()
    assert robust.fall_retry_state(env).still_time.eq(0).all()


def test_source_command_holds_until_goal_has_been_stable():
    env, term = environment()
    term.reset(torch.arange(2))
    for _ in range(800):
        term.compute(env.step_dt)
    torch.testing.assert_close(term.command[:, 0], torch.tensor([1., -1.]))


def test_native_play_keys_queue_and_apply_commands_without_pose_reset():
    from collections import deque

    from mjlab.viewer.base import ViewerAction

    from zbot_rl_mjlab import viewer

    env, term = environment()
    term.reset(torch.arange(2))
    observation_updates = []
    env.observation_manager = SimpleNamespace(compute=lambda **kwargs: observation_updates.append(kwargs))
    instance = object.__new__(viewer.FollowCameraViewer)
    instance.env = cast(RslRlVecEnvWrapper, SimpleNamespace(unwrapped=env))
    instance.env_idx = 0
    instance._actions = deque()
    joints = env.scene["robot"].data.joint_pos.clone()
    other_command = term.command[1].clone()
    for key, expected in ((ord("]"), 1), (ord("["), -1)):
        instance._on_key(key)
        action, payload = instance._actions.popleft()
        assert action == ViewerAction.CUSTOM
        assert instance._handle_custom_action(action, payload)
        assert term.command[0].eq(expected).all()
        torch.testing.assert_close(term.command[1], other_command)
        torch.testing.assert_close(env.scene["robot"].data.joint_pos, joints)
    assert len(observation_updates) == 2
    for update in observation_updates:
        assert update['update_history'] is True
        torch.testing.assert_close(update['env_ids'], torch.tensor([0]))


def test_interactive_switch_can_clear_state_created_during_inference():
    env, term = environment()
    with torch.inference_mode():
        env._standing_goal_valid = torch.ones(2, dtype=torch.bool)
    term.set_target("snake")
    assert not env._standing_goal_valid.any()


def test_partial_reset_handles_goal_mask_from_inference_without_affecting_others(monkeypatch):
    env, _term = environment()
    env.cfg.initial_standing_fraction = 0
    with torch.inference_mode():
        env._standing_goal_valid = torch.ones(2, dtype=torch.bool)

    def clear_goal(env, ids):
        env._standing_goal_valid[ids] = False

    monkeypatch.setattr(robust, "reset_fall_retry", clear_goal)
    commanded.reset_command_transition(env, torch.tensor([0]))
    assert env._standing_goal_valid.tolist() == [False, True]
    assert not mdp.state(env).hold_time.is_inference()
    mdp.state(env).hold_time[:] = 1  # 持久缓冲区仍可由常规调用写入。
    assert env._transform_initial_standing.tolist() == [False, True]


def test_frequent_stand_resets_cannot_starve_live_stand_samples():
    env, term = environment()
    env.num_envs = 128
    term._command = torch.ones(128, 1)
    env._transform_initial_standing = torch.zeros(128, dtype=torch.bool)
    ids = torch.arange(128)
    env._transform_initial_standing[ids] = commanded.sample_initial_standing(env, ids)
    term._command[:, 0] = torch.where(env._transform_initial_standing, -1., 1.)
    assert env._transform_initial_standing.sum() == 64
    # 站立约在 0.12 s 时失败，而下降持续 16 s：复现仅蛇形初始状态
    # 对应的一半环境频繁重置的情况。训练帧必须保持各占 50%。
    for _ in range(150):
        ids = (~env._transform_initial_standing).nonzero().flatten()
        env._transform_initial_standing[ids] = commanded.sample_initial_standing(env, ids)
        term._command[:, 0] = torch.where(env._transform_initial_standing, -1., 1.)
        assert env._transform_initial_standing.sum() == 64
    ids = torch.arange(32)
    original = env._transform_initial_standing.clone()
    env._transform_initial_standing[ids] = commanded.sample_initial_standing(env, ids)
    assert env._transform_initial_standing.sum() == 64
    torch.testing.assert_close(env._transform_initial_standing[32:], original[32:])


def test_balanced_source_assignment_is_random_and_matches_non_half_fraction():
    env, term = environment()
    env.num_envs = 100
    term._command = torch.ones(100, 1)
    env._transform_initial_standing = torch.zeros(100, dtype=torch.bool)
    env.cfg.initial_standing_fraction = 0.3
    ids = torch.arange(100)
    first = commanded.sample_initial_standing(env, ids)
    second = commanded.sample_initial_standing(env, ids)
    assert first.sum() == second.sum() == 30
    assert not torch.equal(first, second)
    for value in (0., 1.):
        env.cfg.initial_standing_fraction = value
        assert commanded.sample_initial_standing(env, ids).sum() == int(100 * value)


def test_population_fraction_reports_time_share_even_for_single_direction_resets():
    env, term = environment()
    term._command[:, 0] = torch.tensor([1., -1.])
    torch.testing.assert_close(commanded.stand_env_fraction(env), torch.tensor([0.5, 0.5]))


def test_single_environment_still_samples_both_sources_over_repeated_resets():
    env, _term = environment()
    env.num_envs = 1
    ids = torch.tensor([0])
    samples = torch.cat([commanded.sample_initial_standing(env, ids) for _ in range(64)])
    assert samples.any() and (~samples).any()



def test_play_keyboard_holds_command_past_timeout_until_reset():
    env, term = environment()
    env.max_episode_length = 800
    env.episode_length_buf[:] = 800
    term.reset(torch.arange(2))
    assert commanded.play_time_out(env).tolist() == [True, True]
    # 重复当前命令时，仍应按设计进入手动模式。
    term.set_target("stand", torch.tensor([0]))
    assert commanded.play_time_out(env).tolist() == [False, True]
    env.episode_length_buf[:] = 10000
    assert commanded.play_time_out(env).tolist() == [False, True]
    term.set_target("snake", torch.tensor([1]))
    assert not commanded.play_time_out(env).any()
    term.reset(torch.tensor([0]))
    assert commanded.play_time_out(env).tolist() == [True, False]


def test_manual_timeout_override_only_applies_to_play():
    training = SnakeBipedBidirectional().build_env()
    assert isinstance(training, BidirectionalEnvCfg)
    playback = SnakeBipedBidirectional().build_env(play=True)
    assert isinstance(playback, BidirectionalEnvCfg)
    assert training.terminations["time_out"].func is commanded.common_mdp.time_out
    assert playback.terminations["time_out"].func is commanded.play_time_out
    assert len(training.rewards) == len(playback.rewards) == 7


@pytest.mark.parametrize("name", ["fallen_after_standing", "early_base_ground_contact", "abandoned_exploration"])
def test_play_manual_commands_mask_every_failure_without_changing_training(name):
    env, term = environment()
    term.reset(torch.arange(2))
    playback = SnakeBipedBidirectional().build_env(play=True)
    assert isinstance(playback, BidirectionalEnvCfg)
    training = SnakeBipedBidirectional().build_env()
    assert isinstance(training, BidirectionalEnvCfg)
    entry = playback.terminations[name]
    assert entry.func is commanded.play_failure
    assert entry.params["condition"] is training.terminations[name].func
    trigger = lambda env: torch.ones(env.num_envs, dtype=torch.bool)
    assert entry.func(env, condition=trigger).tolist() == [True, True]
    term.set_target("stand", torch.tensor([0]))
    assert entry.func(env, condition=trigger).tolist() == [False, True]
    term.set_target("snake", torch.tensor([1]))
    assert not entry.func(env, condition=trigger).any()


def test_bracket_keys_dispatch_custom_commands_without_requesting_viewer_reset():
    from collections import deque

    from mjlab.viewer.base import ViewerAction

    from zbot_rl_mjlab import viewer

    instance = object.__new__(viewer.FollowCameraViewer)
    instance._actions = deque()
    instance.user_key_callback = instance._on_key
    for key, expected in ((ord("]"), "transform_stand"), (ord("["), "transform_snake")):
        instance._safe_key_callback(key)
        action, payload = instance._actions.popleft()
        assert action == ViewerAction.CUSTOM
        assert payload == expected
        assert not instance._actions


def test_initial_roll_contact_separation_and_small_drop_do_not_reset():
    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    contact = env.scene["base_ground_contact"].data.found
    data.body_link_lin_vel_w[:, 6, 2] = -0.05
    for step, height, touching in ((0, 0.0526, True), (1, 0.056, False), (2, 0.052, True), (3, 0.055, True)):
        env.common_step_counter = step
        data.body_link_pos_w[:, 6, 2] = height
        contact[:] = touching
        assert not commanded.early_contact(env).any()
    assert not robust.fall_retry_state(env).early_base_fallen.any()


def test_early_fall_requires_both_height_and_drop_then_latches():
    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    contact = env.scene["base_ground_contact"].data.found
    data.body_link_pos_w[:, 6, 2] = 0.04
    assert not commanded.early_contact(env).any()
    env.common_step_counter += 1
    data.body_link_pos_w[:, 6, 2] = torch.tensor([0.049, 0.065])
    contact[:] = False
    assert not commanded.early_contact(env).any()
    env.common_step_counter += 1
    data.body_link_pos_w[:, 6, 2] = torch.tensor([0.035, 0.060])
    contact[:] = True
    data.body_link_lin_vel_w[:, 6, 2] = -0.05
    assert not commanded.early_contact(env).any()  # 低于启用阈值，下降仅 5 mm。
    env.common_step_counter += 1
    data.body_link_pos_w[:, 6, 2] = 0.035
    assert commanded.early_contact(env).tolist() == [False, True]
    env.common_step_counter += 1
    data.body_link_pos_w[:, 6, 2] = 0.12
    assert commanded.early_contact(env).tolist() == [False, True]
    commanded.clear_command_memory(env, torch.tensor([1]))
    assert not robust.fall_retry_state(env).early_base_fallen.any()


def test_meaningful_soft_contact_drop_still_resets_without_separation():
    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    data.body_link_pos_w[:, 6, 2] = 0.09
    assert not commanded.early_contact(env).any()
    env.common_step_counter += 1
    data.body_link_pos_w[:, 6, 2] = 0.075
    data.body_link_lin_vel_w[:, 6, 2] = -0.05
    assert commanded.early_contact(env).all()


@pytest.mark.parametrize("height", [0., -0.1, float("nan"), float("inf"), 0.3])
def test_invalid_early_fall_arm_height_is_rejected(height):
    env, _term = environment()
    env.cfg.early_contact_arm_height = height
    with pytest.raises(ValueError, match="early_contact_arm_height"):
        commanded.early_contact(env)


def test_low_height_fall_protection_arms_at_five_cm_without_resetting_initial_roll():
    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    for step, height in enumerate((0.0526, 0.055, 0.051, 0.053, 0.0526)):
        env.common_step_counter = step
        data.body_link_pos_w[:, 6, 2] = height
        assert not commanded.fallen(env).any()
        assert not commanded.fallen(env).any()  # 重复调用不会推进时间。
    assert robust.fall_retry_state(env).armed.all()
    assert robust.fall_retry_state(env).low_time.eq(0).all()


def test_five_cm_protection_resets_a_low_fall_after_hold_but_not_mm_drift():
    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    data.body_link_pos_w[:, 6, 2] = torch.tensor([0.065, 0.07])
    assert not commanded.fallen(env).any()
    data.body_link_pos_w[:, 6, 2] = torch.tensor([0.058, 0.0526])
    for step in range(1, 5):
        env.common_step_counter = step
        assert not commanded.fallen(env).any()
        assert not commanded.fallen(env).any()
    env.common_step_counter = 5
    assert commanded.fallen(env).tolist() == [False, True]
    term.set_target("snake", torch.tensor([1]))
    assert not commanded.fallen(env).any()


def test_standing_pose_requires_world_gravity_alignment_not_just_joint_angles():
    from test_transformation_gravity_alignment import fk_environment

    values = []
    for root in ((1., 0., 0., 0.), standing.SNAKE_ROOT_QUAT):
        env = fk_environment(root=root)
        env.cfg = SnakeBipedBidirectional().build_env()
        assert isinstance(env.cfg, BidirectionalEnvCfg)
        env.scene["robot"].data.joint_pos = torch.tensor([env.cfg.standing_joint_pos], dtype=torch.float64)
        env.command_manager = SimpleNamespace(get_command=lambda name: torch.ones(1, 1))
        values.append(commanded.pose_reward(env).item())
    assert values == pytest.approx([0., -1.], abs=1e-9)
    assert 12 * values[1] == pytest.approx(-12.)


def test_pose_orientation_is_yaw_invariant_and_does_not_penalize_descent():
    from mjlab.utils.lab_api.math import quat_mul

    env, term = environment()
    term._command[:, 0] = torch.tensor([1., -1.])
    data = env.scene["robot"].data
    data.joint_pos[1] = 0
    before = commanded.pose_reward(env)
    yaw = torch.tensor([0.5, 0., 0., 3**0.5 / 2]).expand(2, 4)
    data.body_link_quat_w[:, 6] = quat_mul(yaw, data.body_link_quat_w[:, 6])
    torch.testing.assert_close(commanded.pose_reward(env), before, atol=1e-6, rtol=0)
    data.body_link_quat_w[1, 6] = float("nan")
    assert commanded.pose_reward(env)[1] == 0


def test_lying_target_joints_keep_ascent_drive_and_height_latch_still_stops_it():
    from test_transformation_gravity_alignment import fk_environment

    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    data.body_link_pos_w[:, 6, 2] = torch.tensor([0.136, 0.0526])
    lying = fk_environment(root=standing.SNAKE_ROOT_QUAT)
    data.body_link_quat_w[:, 6] = lying.scene["robot"].data.body_link_quat_w[0, 0].float()
    data.body_link_lin_vel_w[:, 6, 2] = 0.02
    full = standing.base_ascent_velocity_reward(env, target_speed=0.15)
    value = commanded.velocity_reward(env)
    assert (value > 0).all()
    assert (value > 0.75 * full).all()  # 躺卧时折叠不能将其归零。
    data.body_link_pos_w[:, 6, 2] = 0.23
    torch.testing.assert_close(commanded.velocity_reward(env), torch.zeros(2))
    data.body_link_pos_w[:, 6, 2] = 0.136
    torch.testing.assert_close(commanded.velocity_reward(env), torch.zeros(2))


def test_spatial_velocity_fade_is_smooth_and_stronger_for_upright_body():
    from test_transformation_gravity_alignment import fk_environment

    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    data.body_link_pos_w[:, 6, 2] = 0.20
    data.body_link_lin_vel_w[:, 6, 2] = 0.02
    lying = fk_environment(root=standing.SNAKE_ROOT_QUAT)
    data.body_link_quat_w[1, 6] = lying.scene["robot"].data.body_link_quat_w[0, 0].float()
    value = commanded.velocity_reward(env)
    assert 0 < value[0] < value[1]
    initial = value[0].item()
    data.body_link_pos_w[:, 6, 2] = 0.21
    assert 0 < commanded.velocity_reward(env)[0] < initial
    data.joint_pos[:, 0] += 0.9
    torch.testing.assert_close(commanded.velocity_reward(env), standing.base_ascent_velocity_reward(env, target_speed=0.15))


@pytest.mark.parametrize("scale", [-1., float("nan"), float("inf")])
def test_pose_gravity_scale_rejects_invalid_values(scale):
    env, _term = environment()
    env.cfg.standing_pose_gravity_scale = scale
    with pytest.raises(ValueError, match="standing_pose_gravity_scale"):
        commanded.pose_reward(env)
