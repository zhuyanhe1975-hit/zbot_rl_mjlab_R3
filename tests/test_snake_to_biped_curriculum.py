"""使用真实 MJLab 管理器、以回合为门控的全局奖励课程。"""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.managers.reward_manager import RewardManager

from zbot_rl_mjlab.tasks.transformation import snake_to_biped as reverse


def _env(play=False, ramp_steps=0, num_envs=2):
    cfg = reverse.snake_to_biped_env_cfg(play=play)
    # 使用固定值测试课程机制，不受实时调参影响。
    cfg.sole_alignment_stage_threshold = 1.0
    cfg.first_stage_joint_weight = 0.1
    cfg.standing_stage_joint_weight = 1.0
    cfg.standing_stage_height_weight = 3.0
    cfg.standing_stage_velocity_weight = 1.0
    for name, weight in zip(
        ("joint_standing", "base_height_progress", "base_ascent_velocity"), (1.0, 3.0, 1.0)
    ):
        cfg.rewards[name].weight = weight if play else (0.1 if name == "joint_standing" else 0.0)
    cfg.base_support_distance_weight = -1.0
    cfg.rewards["base_support_distance"].weight = -1.0
    cfg.standing_stage_ramp_steps = ramp_steps
    env = SimpleNamespace(
        cfg=cfg,
        num_envs=num_envs,
        common_step_counter=0,
        device="cpu",
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        max_episode_length_s=cfg.episode_length_s,
        termination_manager=SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
    )
    # 大多数测试单独检查对齐门控；就绪条件另行测试。
    reverse.support_protection_state(env).support_hold_s[:] = cfg.curriculum_support_hold_s
    env.reward_manager = RewardManager(cfg.rewards, env)
    env.curriculum_manager = CurriculumManager(cfg.curriculum, env)
    return env


def _weights(env):
    return tuple(
        env.reward_manager.get_term_cfg(name).weight
        for name in ("joint_standing", "base_height_progress", "base_ascent_velocity")
    )


@pytest.mark.parametrize("score,stage", [(0.0, 1), (0.999, 1), (1.0, 1), (1.001, 2), (2.0, 2)])
def test_strict_episode_gate_matches_logged_weighted_reward(score, stage):
    env = _env()
    assert _weights(env) == (0.1, 0.0, 0.0)
    env.episode_length_buf[:] = 800
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = score * 16.0
    env.curriculum_manager.compute()
    logs = env.curriculum_manager.reset()
    reward_logs = env.reward_manager.reset()
    assert logs["Curriculum/alignment_to_standing/stage"] == stage
    assert logs["Curriculum/alignment_to_standing/alignment"] == pytest.approx(
        reward_logs["Episode_Reward/sole_ground_alignment"].item()
    )
    assert _weights(env) == ((1.0, 3.0, 1.0) if stage == 2 else (0.1, 0.0, 0.0))
    assert env.reward_manager.get_term_cfg("sole_ground_alignment").weight == 2.0
    assert env.reward_manager.get_term_cfg("reset_failure").weight == -10.0
    assert env.reward_manager.get_term_cfg("base_support_distance").weight == -1.0
    assert env.reward_manager.get_term_cfg("initial_joint_pose").weight == (
        env.cfg.first_stage_initial_pose_weight if stage == 1 else 0.0
    )
    assert env.reward_manager.get_term_cfg("end_joint_flip").weight == env.cfg.end_joint_flip_weight
    assert env.reward_manager.get_term_cfg("axial_roll_rate").weight == (
        env.cfg.first_stage_roll_rate_weight if stage == 1 else 0.0
    )


def test_promotion_uses_average_of_reset_worlds_and_persists_after_partial_resets():
    env = _env()
    env.episode_length_buf[:] = 800
    sums = env.reward_manager._episode_sums["sole_ground_alignment"]
    sums[:] = torch.tensor([0.5, 1.5]) * 16.0
    env.curriculum_manager.compute()
    assert _weights(env) == (0.1, 0.0, 0.0)  # 均值等于 1，并未大于 1。
    env.curriculum_manager.compute(torch.tensor([1]))
    assert _weights(env) == (1.0, 3.0, 1.0)
    env.reward_manager.reset(torch.tensor([1]))
    env.episode_length_buf[1] = 0
    env.curriculum_manager.compute(torch.tensor([1]))
    env.curriculum_manager.compute(torch.tensor([0]))
    assert _weights(env) == (1.0, 3.0, 1.0)
    # 修改运行中的管理器不得改变新建环境配置。
    assert env.cfg.rewards["joint_standing"].weight == 0.1
    assert _weights(_env()) == (0.1, 0.0, 0.0)


def test_initial_reset_and_empty_selection_cannot_promote():
    env = _env()
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 32.0
    env.curriculum_manager.compute()
    env.curriculum_manager.compute(torch.tensor([], dtype=torch.long))
    assert _weights(env) == (0.1, 0.0, 0.0)


def test_play_and_explicit_stage_two_start_with_full_objective():
    play = _env(play=True)
    play.curriculum_manager.compute()
    assert _weights(play) == (1.0, 3.0, 1.0)
    resumed = _env()
    resumed.cfg.curriculum_start_stage = 2
    resumed.cfg.standing_stage_joint_weight = 4.0
    resumed.curriculum_manager.compute()
    assert _weights(resumed) == (4.0, 3.0, 1.0)


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_zero_weight_rewards_become_active_and_gate_is_independent_of_dt(dt):
    env = _env()
    objectives = {"joint_standing", "base_height_progress", "base_ascent_velocity"}
    for name, term in env.cfg.rewards.items():
        value = 1.0 if name in objectives | {"sole_ground_alignment", "base_support_distance"} else 0.0
        term.func = lambda env, value=value: torch.full((env.num_envs,), value)
        term.params = {}
    env.reward_manager = RewardManager(env.cfg.rewards, env)
    steps = round(16.0 / dt)
    for _ in range(steps):
        reward = env.reward_manager.compute(dt)
    torch.testing.assert_close(reward, torch.full((2,), 1.1 * dt))
    env.episode_length_buf[:] = steps
    env.curriculum_manager.compute()
    assert _weights(env) == (1.0, 3.0, 1.0)
    torch.testing.assert_close(env.reward_manager.compute(dt), torch.full((2,), 6.0 * dt))


@pytest.mark.parametrize("threshold", [2.0, -1.0, float("nan")])
def test_unreachable_alignment_threshold_is_rejected(threshold):
    env = _env()
    env.cfg.sole_alignment_stage_threshold = threshold
    with pytest.raises(ValueError, match="threshold"):
        env.curriculum_manager.compute()


def test_promotion_ramps_over_control_steps_and_keeps_progress_across_resets():
    assert reverse.snake_to_biped_env_cfg().standing_stage_ramp_steps == 2400
    env = _env(ramp_steps=2400)
    env.common_step_counter = 100
    env.episode_length_buf[:] = 800
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 32.0
    env.curriculum_manager.compute()
    assert env._standing_curriculum_stage == 2
    assert env.reward_manager.get_term_cfg("initial_joint_pose").weight == 0.0
    assert _weights(env) == (0.1, 0.0, 0.0)
    assert env.reward_manager.get_term_cfg("base_support_distance").weight == -1.0
    env.reward_manager.reset()
    env.episode_length_buf[:] = 0
    for elapsed, scale in ((1, 1 / 2400), (600, 0.25), (1200, 0.5), (2400, 1.0), (3000, 1.0)):
        env.common_step_counter = 100 + elapsed
        assert reverse.apply_standing_stage_weights(env) == pytest.approx(scale)
        assert _weights(env) == pytest.approx((0.1 + 0.9 * scale, 3.0 * scale, scale))
        env.curriculum_manager.compute(torch.tensor([0]))
        assert _weights(env) == pytest.approx((0.1 + 0.9 * scale, 3.0 * scale, scale))
        assert env._standing_stage_start_step == 100
        assert env.reward_manager.get_term_cfg("sole_ground_alignment").weight == 2.0
        assert env.reward_manager.get_term_cfg("base_support_distance").weight == -1.0


def test_playback_skips_ramp_and_negative_ramp_duration_is_rejected():
    env = _env(play=True, ramp_steps=2400)
    env.curriculum_manager.compute()
    assert _weights(env) == (1.0, 3.0, 1.0)
    env.cfg.standing_stage_ramp_steps = -1
    with pytest.raises(ValueError, match="nonnegative"):
        env.curriculum_manager.compute()


def test_initial_pose_penalty_stops_immediately_on_promotion_before_ramp():
    env = _env(ramp_steps=2400)
    for name, term in env.cfg.rewards.items():
        value = 1.0 if name == "initial_joint_pose" else 0.0
        term.func = lambda env, value=value: torch.full((env.num_envs,), value)
        term.params = {}
    env.reward_manager = RewardManager(env.cfg.rewards, env)
    dt = 0.02
    torch.testing.assert_close(
        env.reward_manager.compute(dt), torch.full((2,), env.cfg.first_stage_initial_pose_weight * dt)
    )
    env.episode_length_buf[:] = 800
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 32.0
    env.curriculum_manager.compute()
    assert env._standing_curriculum_stage == 2
    assert _weights(env) == (0.1, 0.0, 0.0)
    torch.testing.assert_close(env.reward_manager.compute(dt), torch.zeros(2))
    env.reward_manager.reset()
    env.curriculum_manager.compute()
    assert env.reward_manager.get_term_cfg("initial_joint_pose").weight == 0.0


@pytest.mark.parametrize("ready_count,promoted", [(0, False), (3, False), (4, True), (5, True)])
def test_promotion_requires_eighty_percent_with_continuous_support(ready_count, promoted):
    env = _env(num_envs=5)
    env.episode_length_buf[:] = 800
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 32.0
    held = reverse.support_protection_state(env).support_hold_s
    held[:] = 0.499
    held[:ready_count] = 0.5
    env.curriculum_manager.compute()
    assert env._standing_curriculum_stage == (2 if promoted else 1)
    logs = env.curriculum_manager.reset()
    assert logs["Curriculum/alignment_to_standing/support_ready_fraction"] == pytest.approx(ready_count / 5)
    # 即使在课程升级瞬间，定向翻转惩罚也应保持启用。
    assert env.reward_manager.get_term_cfg("end_joint_flip").weight == env.cfg.end_joint_flip_weight


def test_failed_or_unfinished_world_cannot_satisfy_support_gate():
    env = _env()
    env.episode_length_buf[:] = 800
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 32.0
    env.termination_manager.terminated[1] = True
    env.curriculum_manager.compute()
    assert env._standing_curriculum_stage == 1
    env.termination_manager.terminated[:] = False
    env.episode_length_buf[:] = 0
    env.curriculum_manager.compute()
    assert env._standing_curriculum_stage == 1


def test_one_ready_reset_batch_cannot_promote_all_worlds_early():
    env = _env(num_envs=5)
    env.episode_length_buf[:] = 800
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 32.0
    env.curriculum_manager.compute(torch.tensor([0]))
    assert env._standing_curriculum_stage == 1
    assert reverse.support_protection_state(env).last_episode_ready.tolist() == [True, False, False, False, False]
    env.curriculum_manager.compute(torch.tensor([1, 2, 3]))
    assert env._standing_curriculum_stage == 2
    assert env.reward_manager.get_term_cfg("end_joint_flip").weight == -1.0


@pytest.mark.parametrize("field,value", [
    ("curriculum_support_hold_s", 0.0), ("curriculum_support_hold_s", float("nan")),
    ("curriculum_support_ready_fraction", 0.0), ("curriculum_support_ready_fraction", 1.1),
])
def test_invalid_support_curriculum_parameters_are_rejected(field, value):
    env = _env()
    setattr(env.cfg, field, value)
    with pytest.raises(ValueError, match=field):
        env.curriculum_manager.compute()


def test_hot_alignment_change_waits_full_episode_but_updates_support_history(tmp_path):
    import json

    from zbot_rl_mjlab.rl import HotRewardController

    env = _env()
    env.max_episode_length = 800
    env.common_step_counter = 100
    path = tmp_path / "weights.json"
    env.cfg.reward_hot_reload_path = str(path)
    path.write_text(json.dumps({"rewards": {"sole_ground_alignment": 3.0}}))
    assert HotRewardController(env).poll(1)
    env.episode_length_buf[:] = 800
    # 即使得分满足其他条件，只要仍可能有混合权重的回合
    # 尚未结束，就不得升级。
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 32.0
    logs = reverse.alignment_to_standing_curriculum(env, None)
    assert logs["stage"] == 1
    assert logs["alignment_hot_wait_steps"] == 800
    assert logs["support_ready_fraction"] == 1.0
    assert reverse.support_protection_state(env).last_episode_ready.all()
    env.common_step_counter = 899
    assert reverse.alignment_to_standing_curriculum(env, None)["stage"] == 1
    env.common_step_counter = 900
    logs = reverse.alignment_to_standing_curriculum(env, None)
    assert logs["alignment_hot_wait_steps"] == 0
    assert logs["stage"] == 2


def test_hot_alignment_wait_never_demotes_existing_stage_two():
    env = _env(play=True)
    env.max_episode_length = 800
    env._standing_curriculum_stage = 2
    env._standing_alignment_hot_changed_step = 0
    logs = reverse.alignment_to_standing_curriculum(env, None)
    assert logs["stage"] == 2
    assert logs["alignment_hot_wait_steps"] == 800
    assert _weights(env) == (1.0, 3.0, 1.0)


def _pose_gate_env(num_envs=5, threshold=1.0):
    env = _env(num_envs=num_envs)
    env.cfg.curriculum_joint_error_threshold = threshold
    target = torch.tensor(env.cfg.standing_joint_pos)
    positions = target.expand(num_envs, -1).clone()
    env.scene = {"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=positions))}
    env.episode_length_buf[:] = 800
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 32.0
    return env


def test_r3_wrong_fold_cannot_promote_despite_aligned_supported_feet():
    env = _pose_gate_env()
    # R3/model_1198 的实测终止姿态；q3/q4 仍处于错误分支。
    env.scene["robot"].data.joint_pos[:] = torch.tensor(
        [1.298926, -0.762142, 0.272485, -0.189076, 0.713332, -1.356197]
    )
    logs = reverse.alignment_to_standing_curriculum(env, None)
    assert logs["stage"] == 1
    assert logs["support_ready_fraction"] == 0
    assert logs["joint_error_threshold"] == 1.0


def test_coarse_pose_and_support_must_qualify_together_at_episode_end():
    env = _pose_gate_env()
    positions = env.scene["robot"].data.joint_pos
    positions += 0.5
    reverse.support_protection_state(env).support_hold_s[:] = 0.0
    assert reverse.alignment_to_standing_curriculum(env, None)["stage"] == 1
    # 仅接近目标姿态仍不足够；还必须满足支撑条件。
    reverse.support_protection_state(env).support_hold_s[:] = env.cfg.curriculum_support_hold_s
    assert reverse.alignment_to_standing_curriculum(env, None)["stage"] == 2
    # 一旦升级，即使后续出现严重折叠姿态，也不应降级。
    positions += 3.0
    assert reverse.alignment_to_standing_curriculum(env, None)["stage"] == 2


def test_coarse_pose_gate_counts_latest_completed_outcomes_over_all_worlds():
    env = _pose_gate_env()
    support = reverse.support_protection_state(env)
    reverse.alignment_to_standing_curriculum(env, torch.tensor([0]))
    assert env._standing_curriculum_stage == 1
    assert support.last_episode_ready.tolist() == [True, False, False, False, False]
    # 环境 0 以不良姿态再次结束时，会覆盖先前的就绪结果。
    env.scene["robot"].data.joint_pos[0] += 2.0
    reverse.alignment_to_standing_curriculum(env, torch.tensor([0, 1, 2, 3]))
    assert support.last_episode_ready.tolist() == [False, True, True, True, False]
    assert env._standing_curriculum_stage == 1
    logs = reverse.alignment_to_standing_curriculum(env, torch.tensor([4]))
    assert logs["support_ready_fraction"] == pytest.approx(0.8)
    assert env._standing_curriculum_stage == 2


def test_disabled_coarse_pose_gate_preserves_original_promotion_behavior():
    env = _pose_gate_env(threshold=None)
    env.scene["robot"].data.joint_pos[:] = 10.0
    logs = reverse.alignment_to_standing_curriculum(env, None)
    assert logs["stage"] == 2
    assert logs["joint_error_threshold"] == -1.0


def test_pose_gate_is_strict_and_does_not_count_unfinished_or_failed_worlds():
    env = _pose_gate_env(threshold=1.0)
    positions = env.scene["robot"].data.joint_pos
    positions[0, 0] += 1.0
    env.termination_manager.terminated[1] = True
    env.episode_length_buf[2] = 0
    reverse.alignment_to_standing_curriculum(env, None)
    assert reverse.support_protection_state(env).last_episode_ready.tolist() == [False, False, False, True, True]
    assert env._standing_curriculum_stage == 1


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf"), True, False])
def test_invalid_coarse_pose_threshold_fails_before_curriculum_mutation(threshold):
    env = _pose_gate_env(threshold=threshold)
    with pytest.raises(ValueError, match="curriculum_joint_error_threshold"):
        reverse.alignment_to_standing_curriculum(env, None)
    assert not hasattr(env, "_standing_curriculum_stage")


@pytest.mark.parametrize("stage", [1, 2])
def test_experimental_full_gravity_weight_persists_in_both_stages(stage):
    env = _env()
    assert env.cfg.base_gravity_alignment_weight == 0.0
    env.cfg.base_gravity_alignment_weight = -0.75
    env._standing_curriculum_stage = stage
    env._standing_stage_start_step = env.common_step_counter
    for step in (0, 100, 5000):
        env.common_step_counter = step
        reverse.apply_standing_stage_weights(env)
        assert env.reward_manager.get_term_cfg("base_gravity_alignment").weight == -0.75


@pytest.mark.parametrize("stage", [1, 2])
def test_optional_survival_weight_remains_fixed_across_stages_and_ramp(stage):
    env = _env(ramp_steps=2400)
    assert env.cfg.survival_weight == 0.0
    env.cfg.survival_weight = 4.0
    env._standing_curriculum_stage = stage
    env._standing_stage_start_step = 0
    for step in (0, 1200, 2400, 5000):
        env.common_step_counter = step
        reverse.apply_standing_stage_weights(env)
        assert env.reward_manager.get_term_cfg("survival").weight == 4.0


def _automatic_prepare_env():
    env = _pose_gate_env()
    env.cfg.curriculum_prepare_pose = True
    env.cfg.first_stage_joint_weight = 0.6
    env.cfg.first_stage_prepared_joint_weight = 3.0
    env.cfg.standing_stage_joint_weight = 6.0
    env.cfg.standing_stage_ramp_steps = 100
    env._standing_curriculum_stage = 1
    reverse.apply_standing_stage_weights(env)
    return env


def test_automatic_pose_preparation_latches_after_alignment_without_arming_guard():
    env = _automatic_prepare_env()
    env.scene["robot"].data.joint_pos += 2.0
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 0.6
    logs = reverse.alignment_to_standing_curriculum(env, None)
    assert logs["pose_prepared"] == 1.0 and logs["stage"] == 1
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 3.0
    assert not reverse.support_protection_state(env).support_armed.any()
    env.reward_manager._episode_sums["sole_ground_alignment"][:] = 0.0
    logs = reverse.alignment_to_standing_curriculum(env, None)
    assert logs["pose_prepared"] == 1.0 and logs["stage"] == 1
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 3.0


def test_preparation_waits_alignment_cooldown_and_completed_episode():
    env = _automatic_prepare_env()
    env.scene["robot"].data.joint_pos += 2.0
    env.max_episode_length = 800
    env._standing_alignment_hot_changed_step = 0
    env.common_step_counter = 799
    assert reverse.alignment_to_standing_curriculum(env, None)["pose_prepared"] == 0.0
    env.common_step_counter = 800
    env.episode_length_buf[:] = 0
    assert reverse.alignment_to_standing_curriculum(env, None)["pose_prepared"] == 0.0
    env.episode_length_buf[:] = 800
    assert reverse.alignment_to_standing_curriculum(env, None)["pose_prepared"] == 1.0


def test_auto_course_waits_global_pose_support_gate_then_ramps_from_three():
    env = _automatic_prepare_env()
    env.scene["robot"].data.joint_pos += 2.0
    reverse.alignment_to_standing_curriculum(env, None)
    env.scene["robot"].data.joint_pos[:4] = torch.tensor(env.cfg.standing_joint_pos)
    logs = reverse.alignment_to_standing_curriculum(env, torch.tensor([0]))
    assert logs["stage"] == 1 and logs["support_ready_fraction"] == pytest.approx(0.2)
    logs = reverse.alignment_to_standing_curriculum(env, torch.tensor([1, 2, 3]))
    assert logs["stage"] == 2 and logs["support_ready_fraction"] == pytest.approx(0.8)
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 3.0
    env.common_step_counter = 50
    reverse.apply_standing_stage_weights(env)
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 4.5
    env.common_step_counter = 100
    reverse.apply_standing_stage_weights(env)
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 6.0


def test_auto_stage_two_missing_old_preparation_flag_still_uses_prepared_baseline():
    env = _automatic_prepare_env()
    env._standing_curriculum_stage = 2
    env._standing_stage_start_step = 0
    assert not hasattr(env, "_standing_pose_prepared")
    reverse.apply_standing_stage_weights(env)
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 3.0
    env.common_step_counter = 50
    reverse.apply_standing_stage_weights(env)
    assert env.reward_manager.get_term_cfg("joint_standing").weight == 4.5
