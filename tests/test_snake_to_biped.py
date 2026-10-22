"""反向变形目标、触地站立成功条件与隔离性。"""

import math
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.experiments.transformation import (
    snake_to_biped_env_cfg as _recipe_snake_to_biped_env_cfg,
)
from zbot_rl_mjlab.robot import DEFAULT_JOINT_POS, JOINT_NAMES
from zbot_rl_mjlab.tasks.transformation import mdp
from zbot_rl_mjlab.tasks.transformation import snake_to_biped as reverse
from zbot_rl_mjlab.tasks.transformation.poses import SNAKE_ROOT_QUAT


def _standing_env(num_envs=2):
    class Scene(dict):
        pass

    positions = torch.zeros(num_envs, 12, 3)
    positions[..., 2] = 0.158
    data = SimpleNamespace(
        joint_pos=torch.tensor(DEFAULT_JOINT_POS).expand(num_envs, 6).clone(),
        joint_vel=torch.zeros(num_envs, 6),
        body_com_pos_w=positions,
        body_com_lin_vel_w=torch.zeros(num_envs, 12, 3),
        body_link_pos_w=positions.clone(),
        body_link_lin_vel_w=torch.zeros(num_envs, 12, 3),
        body_link_ang_vel_w=torch.zeros(num_envs, 12, 3),
        body_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0])
        .expand(num_envs, 12, 4)
        .clone(),
        root_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_envs, 4).clone(),
        site_pos_w=torch.zeros(num_envs, 2, 3),
        site_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_envs, 2, 4).clone(),
    )
    data.body_link_pos_w[:, 6, 2] = 0.264
    data.site_quat_w[:, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    data.body_link_quat_w[:, 6] = torch.tensor(
        [-0.3739189506, 0.3735469609, -0.6002646992, -0.6002745849]
    )
    scene = Scene(
        robot=SimpleNamespace(
            data=data,
            find_bodies=lambda name: ([6], [name]),
            find_sites=lambda names, preserve_order=True: ([0, 1], names),
            indexing=SimpleNamespace(body_ids=torch.arange(12)),
        )
    )
    scene.env_origins = torch.zeros(num_envs, 3)
    scene["feet_ground_contact"] = SimpleNamespace(data=SimpleNamespace(found=torch.ones(num_envs, 2)))
    cfg = _recipe_snake_to_biped_env_cfg(play=False)
    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        cfg=cfg,
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(12))),
        step_dt=0.02,
        common_step_counter=0,
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(num_envs, dtype=torch.bool)
        ),
    )


@pytest.mark.parametrize("play", [False, True])
def test_registered_reverse_task_adds_sole_alignment_and_keeps_independent_actions(
    play,
):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from zbot_rl_mjlab.tasks.transformation.actions import JointTargetRateActionCfg

    cfg = load_env_cfg(reverse.TASK_ID, play=play)
    assert isinstance(cfg, reverse.SnakeToBipedEnvCfg)
    assert [sensor.name for sensor in cfg.scene.sensors] == ["feet_ground_contact"]
    assert isinstance(cfg.actions["joint_pos"], JointTargetRateActionCfg)
    assert cfg.actions["joint_pos"].actuator_names == (".*",)
    assert cfg.actions["joint_pos"].joint_target_bounds == {
        JOINT_NAMES[i]: (cfg.standing_joint_pos[i] - cfg.end_joint_allowed_error,
                         cfg.standing_joint_pos[i] + cfg.end_joint_allowed_error)
        for i in (0, 5)
    }
    assert 0.0 < cfg.base_z_roll_limit_deg < 180.0
    assert cfg.terminations["base_z_roll"].func is reverse.base_z_roll_exceeded
    assert not cfg.terminations["base_z_roll"].time_out
    assert {term.func: term.weight for term in cfg.rewards.values()} == {
        reverse.end_joint_flip_cost: cfg.end_joint_flip_weight,
        reverse.axial_roll_rate_cost: 0.0 if play else cfg.first_stage_roll_rate_weight,
        reverse.initial_joint_pose_cost: 0.0 if play else cfg.first_stage_initial_pose_weight,
        reverse.joint_standing_reward: cfg.standing_stage_joint_weight if play else cfg.first_stage_joint_weight,
        reverse.base_height_progress_reward: cfg.standing_stage_height_weight if play else 0.0,
        reverse.base_ascent_velocity_reward: cfg.standing_stage_velocity_weight if play else 0.0,
        reverse.base_support_distance_cost: cfg.base_support_distance_weight,
        mdp.joint_symmetry_cost: -1.0,
        reverse.sole_ground_alignment_reward: 2.0,
        reverse.both_feet_contact_metric: cfg.feet_contact_weight,
        reverse.sole_height_cost: -1.0,
        reverse.base_z_roll_cost: -1.0,
        reverse.base_gravity_alignment_cost: 0.0,
        reverse.reset_failure_cost: -10.0,
        reverse.survival_reward: 0.0,
    }
    assert cfg.rewards["base_ascent_velocity"].params == {"target_speed": 0.1}
    assert cfg.curriculum_start_stage == (2 if play else 1)
    assert cfg.curriculum["alignment_to_standing"].func is reverse.alignment_to_standing_curriculum
    assert list(cfg.observations["actor"].terms) == [
        "joint_pos",
        "joint_vel",
        "commanded_target",
        "last_action",
        "com_state",
        "projected_gravity",
        "hold_time",
        "support_state",
    ]
    assert list(cfg.observations["critic"].terms) == list(
        cfg.observations["actor"].terms
    )
    assert (
        load_rl_cfg(reverse.TASK_ID).experiment_name == "zbot_transform_snake_to_biped"
    )


def test_hot_reward_runner_is_registered_only_for_reverse_transformation():
    from mjlab.tasks.registry import load_runner_cls

    from zbot_rl_mjlab.rl import SnakeToBipedHotRewardRunner
    from zbot_rl_mjlab.tasks.transformation.env_cfg import TASK_ID as forward_id

    assert load_runner_cls(reverse.TASK_ID) is SnakeToBipedHotRewardRunner
    assert load_runner_cls(forward_id) is None


@pytest.mark.parametrize("reverse_first", [False, True])
def test_constructing_reverse_does_not_mutate_forward_pose_or_objectives(reverse_first):
    from zbot_rl_mjlab.experiments.transformation import transformation_env_cfg

    factories = [transformation_env_cfg, _recipe_snake_to_biped_env_cfg]
    if reverse_first:
        factories.reverse()
    configs = {factory.__name__: factory(play=True) for factory in factories}
    forward = configs["transformation_env_cfg"]
    backward = configs["snake_to_biped_env_cfg"]
    assert forward.scene.entities["robot"].init_state.joint_pos == dict(
        zip(JOINT_NAMES, DEFAULT_JOINT_POS)
    )
    assert {term.func for term in forward.rewards.values()} == {
        mdp.joint_zero_reward,
        mdp.com_descent_velocity_reward,
        mdp.joint_symmetry_cost,
    }
    assert forward.terminations["update_state"].func is mdp.update_state
    assert backward.terminations["update_state"].func is reverse.update_state
    assert forward.metrics["success"].func is mdp.success
    assert backward.metrics["success"].func is reverse.success
    assert "base_z_roll" not in forward.terminations
    assert "base_z_roll" not in forward.rewards
    assert "reset_failure" not in forward.rewards
    backward.rewards["joint_symmetry"].weight = -2.0
    assert forward.rewards["joint_symmetry"].weight == -0.5


def test_reverse_ppo_and_control_period_match_original_transformation_defaults():
    from dataclasses import asdict

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from zbot_rl_mjlab.experiments.common import ppo_cfg
    from zbot_rl_mjlab.tasks.transformation.env_cfg import TASK_ID as forward_id

    cfg = load_env_cfg(reverse.TASK_ID)
    reverse_ppo = load_rl_cfg(reverse.TASK_ID)
    forward = load_env_cfg(forward_id)
    forward_ppo = load_rl_cfg(forward_id)
    dt = cfg.decimation * cfg.sim.mujoco.timestep
    old_dt = forward.decimation * forward.sim.mujoco.timestep
    assert cfg.sim.mujoco.timestep == forward.sim.mujoco.timestep == 0.005
    assert dt == old_dt == pytest.approx(0.02)
    assert dt * reverse_ppo.num_steps_per_env == pytest.approx(0.48)
    assert reverse_ppo.actor.distribution_cfg["std_range"] == (0.1, 1.0)
    assert reverse_ppo.algorithm.use_clipped_value_loss
    assert reverse_ppo.algorithm.gamma == 0.99
    assert reverse_ppo.algorithm.lam == 0.95
    assert reverse_ppo.algorithm.num_mini_batches == 4
    assert load_env_cfg(reverse.TASK_ID, play=True).decimation == cfg.decimation
    assert forward.decimation == 4
    assert forward_ppo.num_steps_per_env == ppo_cfg().num_steps_per_env == 24
    assert forward_ppo.actor.distribution_cfg["std_range"] == (0.1, 1.0)
    assert forward_ppo.algorithm.use_clipped_value_loss
    reverse_settings = asdict(reverse_ppo)
    forward_settings = asdict(forward_ppo)
    reverse_settings.pop("experiment_name")
    forward_settings.pop("experiment_name")
    # The forward best has its own archived 200-iteration launch schedule.
    assert forward_settings.pop("max_iterations") == 200
    assert reverse_settings.pop("max_iterations") == 1000
    assert reverse_settings == forward_settings


def test_reverse_config_builders_are_isolated_from_every_other_experiment():
    from dataclasses import asdict

    import yaml
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from zbot_rl_mjlab import EXPERIMENTS
    from zbot_rl_mjlab.experiments.transformation import SnakeToBiped

    others = [cls for cls in EXPERIMENTS if cls is not SnakeToBiped]

    def snapshot():
        return {
            cls.task_id: (
                asdict(cls().build_ppo()),
                asdict(load_rl_cfg(cls.task_id)),
                yaml.dump(asdict(load_env_cfg(cls.task_id)), sort_keys=True),
                yaml.dump(asdict(load_env_cfg(cls.task_id, play=True)), sort_keys=True),
            )
            for cls in others
        }

    before = snapshot()
    tuned = SnakeToBiped().build_ppo()
    tuned.actor.distribution_cfg["std_range"] = (0.8, 1.0)
    tuned.algorithm.gamma = 0.9
    env = SnakeToBiped().build_env()
    env.decimation = 32
    env.actions["joint_pos"].max_target_speed = 0.2
    assert snapshot() == before
    assert SnakeToBiped().build_ppo().actor.distribution_cfg["std_range"] == (0.1, 1.0)


def test_reverse_initial_pose_is_horizontal_with_all_six_joint_targets_zero():
    from mjlab.utils.lab_api.math import quat_apply_inverse

    pose = _recipe_snake_to_biped_env_cfg().scene.entities["robot"].init_state
    assert pose.joint_pos == dict.fromkeys(JOINT_NAMES, 0.0)
    assert pose.joint_vel == {".*": 0.0}
    gravity = quat_apply_inverse(
        torch.tensor([pose.rot]), torch.tensor([[0.0, 0.0, -1.0]])
    )
    assert gravity[0, 2].abs() < 1e-6
    assert pose.pos[2] > 0


def test_standing_pose_scores_higher_than_snake_and_each_joint_affects_goal():
    env = _standing_env()
    env.scene["robot"].data.joint_pos[1] = 0
    reward = reverse.joint_standing_reward(env)
    assert reward[0] == 0.0
    expected_snake = -torch.nn.functional.smooth_l1_loss(torch.zeros(6), torch.tensor(DEFAULT_JOINT_POS), beta=env.cfg.joint_standing_error_scale)
    torch.testing.assert_close(reward[1], expected_snake)
    assert reverse.joint_standing_error_metric(env)[0] == 0
    assert reverse.final_condition(env).tolist() == [True, False]


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("displacement", [-0.3, 0.3])
def test_initial_joint_pose_cost_counts_each_joint_and_both_directions(joint, displacement):
    env = _standing_env()
    env.scene["robot"].data.joint_pos[:] = torch.tensor(reverse.SNAKE_JOINT_POS)
    env.scene["robot"].data.joint_pos[1, joint] += displacement
    torch.testing.assert_close(
        reverse.initial_joint_pose_cost(env), torch.tensor([0.0, displacement**2 / 6.0])
    )


def test_initial_joint_pose_cost_is_independent_of_world_pose_and_finite_on_invalid_joints():
    env = _standing_env()
    env.scene["robot"].data.joint_pos[:] = torch.tensor(reverse.SNAKE_JOINT_POS)
    env.scene["robot"].data.body_link_pos_w[:] = 10.0
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    torch.testing.assert_close(reverse.initial_joint_pose_cost(env), torch.zeros(2))
    env.scene["robot"].data.joint_pos[0, 0] = float("nan")
    env.scene["robot"].data.joint_pos[1, 0] = float("inf")
    torch.testing.assert_close(reverse.initial_joint_pose_cost(env), torch.full((2,), 100.0))


@pytest.mark.parametrize("joint", [0, 5])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_end_joint_flip_penalty_has_deadband_and_targets_only_end_joints(joint, sign):
    env = _standing_env()
    q = env.scene["robot"].data.joint_pos
    q[:, 1:5] = 4.0  # 中间关节的弯曲不属于此惩罚项的约束范围。
    q[0, joint] += sign * 0.9
    q[1, joint] += sign * 2.0
    torch.testing.assert_close(reverse.end_joint_flip_cost(env), torch.tensor([0.0, 0.5]))


def test_end_joint_flip_penalty_rejects_observed_pi_shortcut_but_accepts_snake_and_target():
    env = _standing_env()
    env.scene["robot"].data.joint_pos[0] = torch.tensor(reverse.SNAKE_JOINT_POS)
    torch.testing.assert_close(reverse.end_joint_flip_cost(env), torch.zeros(2))
    env.scene["robot"].data.joint_pos[1, 0] = 3.087
    env.scene["robot"].data.joint_pos[1, 5] = -3.063
    assert reverse.end_joint_flip_cost(env)[1] > 3.0


def test_roll_rate_penalty_uses_body_axis_in_world_frame_not_world_z():
    env = _standing_env()
    data = env.scene["robot"].data
    data.body_link_quat_w[:, 6] = torch.tensor([0.5**0.5, 0.0, 0.5**0.5, 0.0])
    data.body_link_ang_vel_w[0, 6] = torch.tensor([2.0, 0.0, 0.0])
    data.body_link_ang_vel_w[1, 6] = torch.tensor([0.0, 0.0, 2.0])
    torch.testing.assert_close(reverse.axial_roll_rate_cost(env), torch.tensor([4.0, 0.0]), atol=1e-6, rtol=1e-6)
    data.body_link_ang_vel_w[0, 6] = float("nan")
    assert reverse.axial_roll_rate_cost(env)[0] == 100.0


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_each_joint_improvement_has_signal_beyond_old_clipped_plateau(joint, sign):
    env = _standing_env()
    target = torch.tensor(DEFAULT_JOINT_POS)
    positions = env.scene["robot"].data.joint_pos
    positions[:] = target + sign * 4.0
    positions[1, joint] -= sign * 0.1
    # 旧的全姿态均方误差截断无法区分这两种状态。
    old = (1.0 - (positions - target).square().mean(-1) / target.square().mean()).clamp(-1.0, 1.0)
    torch.testing.assert_close(old, -torch.ones(2))
    reward = reverse.joint_standing_reward(env)
    assert reward[0] < reward[1] < 0.0
    # 远离目标时，任意关节修正 0.1 rad 都具有相同的线性增益。
    assert (reward[1] - reward[0]).item() == pytest.approx(0.1 / 6.0, abs=1e-6)


def test_joint_penalty_uses_quadratic_near_target_and_linear_far_error():
    env = _standing_env()
    positions = env.scene["robot"].data.joint_pos
    positions[0] += 0.5
    positions[1] += 1.0
    reward = reverse.joint_standing_reward(env)
    torch.testing.assert_close(reward, torch.tensor([-0.25, -0.75]))
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = 0.1001
    torch.testing.assert_close(reverse.joint_standing_reward(env), reward)


@pytest.mark.parametrize("scale", [0.0, -0.5, float("nan"), float("inf")])
def test_joint_proximity_rejects_invalid_error_scale(scale):
    env = _standing_env()
    env.cfg.joint_standing_error_scale = scale
    with pytest.raises(ValueError, match="finite and positive"):
        reverse.joint_standing_reward(env)


def test_lying_target_pose_gets_less_combined_pose_and_height_reward_than_standing():
    env = _standing_env()
    env.scene["robot"].data.body_link_pos_w[1, 6, 2] = 0.061
    reward = 3.0 * reverse.joint_standing_reward(env) + 3.0 * reverse.base_height_progress_reward(env)
    assert reward[0] == 3.0
    assert reward[1] < 0.3


@pytest.mark.parametrize("height", [0.10, 0.12, 0.14])
def test_standing_reward_increases_smoothly_with_height_toward_standing(height):
    env = _standing_env()
    env.scene["robot"].data.body_link_pos_w[0, 6, 2] = height
    env.scene["robot"].data.body_link_pos_w[1, 6, 2] = height + 0.01
    reward = reverse.base_height_progress_reward(env)
    assert 0 < reward[0] < reward[1] < 1.0


def test_height_progress_saturates_at_standing_height():
    env = _standing_env()
    env.scene["robot"].data.body_link_pos_w[1, 6, 2] = 0.30
    torch.testing.assert_close(reverse.base_height_progress_reward(env), torch.ones(2))


def test_base_height_reward_and_success_ignore_com_height_and_body_masses():
    env = _standing_env()
    data = env.scene["robot"].data
    data.body_com_pos_w[1, :, 2] = 0.5
    env.sim.model.body_mass[0] = 100.0
    torch.testing.assert_close(reverse.base_height_progress_reward(env), torch.ones(2))
    assert reverse.final_condition(env).all()
    data.body_link_pos_w[1, 6, 2] = reverse.SNAKE_ROOT_POS[2]
    torch.testing.assert_close(reverse.base_height_progress_reward(env), torch.tensor([1.0, 0.0]))
    assert reverse.final_condition(env).all()


def test_base_height_reward_uses_ground_origin_and_handles_nonfinite_positions():
    env = _standing_env()
    env.scene.env_origins[1, 2] = 3.0
    env.scene["robot"].data.body_link_pos_w[1, 6, 2] += 3.0
    torch.testing.assert_close(reverse.base_height_progress_reward(env), torch.ones(2))
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = float("nan")
    torch.testing.assert_close(reverse.base_height_progress_reward(env), -torch.ones(2))
    assert not reverse.final_condition(env).any()


@pytest.mark.parametrize("foot", [0, 1])
def test_pose_reward_is_not_gated_by_sole_direction_or_height(foot):
    env = _standing_env()
    data = env.scene["robot"].data
    data.site_quat_w[1, foot] = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
    torch.testing.assert_close(
        reverse.joint_standing_reward(env), torch.zeros(2)
    )
    data.site_quat_w[1, foot] = data.site_quat_w[0, foot]
    data.site_pos_w[1, foot, 2] = 0.06
    torch.testing.assert_close(reverse.joint_standing_reward(env), torch.zeros(2))


def test_standing_reward_is_invariant_to_environment_origin():
    env = _standing_env()
    offset = torch.tensor([10.0, -4.0, 3.0])
    env.scene.env_origins[1] = offset
    env.scene["robot"].data.body_com_pos_w[1] += offset
    env.scene["robot"].data.body_link_pos_w[1] += offset
    env.scene["robot"].data.site_pos_w[1] += offset
    torch.testing.assert_close(reverse.joint_standing_reward(env), torch.zeros(2))


@pytest.mark.parametrize("joint", range(6))
def test_every_joint_must_reach_standing_target(joint):
    env = _standing_env()
    env.scene["robot"].data.joint_pos[1, joint] += env.cfg.joint_tolerance + 0.01
    assert reverse.final_condition(env).tolist() == [True, False]


@pytest.mark.parametrize(
    "speed,expected",
    [(0.0, 0.0), (0.015, 0.75), (0.03, 1.0), (-0.03, -3.0), (0.12, 1.0)],
)
@pytest.mark.parametrize("target_speed", [0.03, 0.1])
def test_ascent_reward_saturates_at_target_speed(speed, expected, target_speed):
    env = _standing_env()
    env.scene["robot"].data.body_link_lin_vel_w[:, 6, 2] = speed / 0.03 * target_speed
    torch.testing.assert_close(
        reverse.base_ascent_velocity_reward(env, target_speed), torch.full((2,), expected)
    )


def test_ascent_reward_uses_base_origin_world_z_instead_of_com_velocity():
    env = _standing_env()
    env.sim.model.body_mass[0] = 11
    env.scene["robot"].data.body_com_lin_vel_w[0, 0, 2] = 0.06
    env.scene["robot"].data.body_link_lin_vel_w[1, 6, 2] = 0.03
    torch.testing.assert_close(
        reverse.base_ascent_velocity_reward(env), torch.tensor([0.0, 1.0])
    )
    # 改变朝向、世界坐标系中的 XY 速度或质心运动时，不得将
    # 所需的世界坐标系 Z 轴信号旋转为局部坐标系分量。
    env.scene["robot"].data.body_link_quat_w[:] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    env.scene["robot"].data.body_link_lin_vel_w[..., :2] = 10.0
    env.scene["robot"].data.body_com_lin_vel_w[:] = -100.0
    torch.testing.assert_close(
        reverse.base_ascent_velocity_reward(env), torch.tensor([0.0, 1.0])
    )


@pytest.mark.parametrize("velocity", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_base_vertical_velocity_gets_finite_negative_reward(velocity):
    env = _standing_env()
    env.scene["robot"].data.body_link_lin_vel_w[:, 6, 2] = velocity
    torch.testing.assert_close(reverse.base_ascent_velocity_reward(env), torch.full((2,), -4.0))


@pytest.mark.parametrize("target_speed", [0.0, -0.03])
def test_ascent_reward_rejects_nonpositive_target_speed(target_speed):
    with pytest.raises(ValueError, match="positive"):
        reverse.base_ascent_velocity_reward(_standing_env(), target_speed)


@pytest.mark.parametrize("height", [0.04, 0.3])
def test_success_has_no_independent_base_height_target(height):
    env = _standing_env()
    env.scene["robot"].data.body_link_pos_w[1, 6, 2] = height
    assert reverse.final_condition(env).all()


@pytest.mark.parametrize("foot,height", [(0, 0.02), (1, 0.02), (0, -0.02), (1, -0.02)])
def test_both_soles_must_be_near_ground_without_penetration(foot, height):
    env = _standing_env()
    env.scene["robot"].data.site_pos_w[1, foot, 2] = height
    assert reverse.final_condition(env).tolist() == [True, False]


@pytest.mark.parametrize("foot", [0, 1])
def test_success_and_support_require_real_contact_even_when_sites_are_near_ground(foot):
    env = _standing_env()
    assert reverse.final_condition(env).all()
    env.scene["feet_ground_contact"].data.found[1, foot] = 0
    assert reverse.both_feet_ground_contact(env).tolist() == [True, False]
    torch.testing.assert_close(reverse.both_feet_contact_metric(env), torch.tensor([1.0, 0.0]))
    assert reverse.sole_support_valid(env).tolist() == [True, False]
    assert reverse.final_condition(env).tolist() == [True, False]


def test_train_and_play_disable_independent_height_and_ascent_objectives():
    for play in (False, True):
        cfg = _recipe_snake_to_biped_env_cfg(play=play)
        assert cfg.standing_stage_height_weight == 0.0
        assert cfg.standing_stage_velocity_weight == 0.0
        assert cfg.rewards["base_height_progress"].weight == 0.0
        assert cfg.rewards["base_ascent_velocity"].weight == 0.0
        assert "base_height" in cfg.metrics and "base_vertical_velocity" in cfg.metrics
        assert cfg.scene.sensors[0].primary.pattern == ("foot_0", "foot_1")


def test_standing_goal_and_safety_use_each_environment_origin():
    env = _standing_env()
    offset = torch.tensor([10.0, -4.0, 3.0])
    env.scene.env_origins[1] = offset
    data = env.scene["robot"].data
    for name in ("body_com_pos_w", "body_link_pos_w", "site_pos_w"):
        getattr(data, name)[1] += offset
    assert reverse.final_condition(env).all()
    assert not mdp.unsafe(env).any()


def test_standing_success_requires_low_joint_and_base_velocity():
    env = _standing_env()
    env.scene["robot"].data.joint_vel[0] = 1
    env.scene["robot"].data.body_link_lin_vel_w[1, 6, 0] = 0.03
    assert not reverse.final_condition(env).any()


def test_standing_success_requires_continuous_two_second_hold():
    env = _standing_env()
    for step in range(99):
        env.common_step_counter = step
        reverse.advance(env)
    assert not reverse.success(env).any()
    env.common_step_counter = 99
    reverse.advance(env)
    assert reverse.success(env).all()
    env.scene["robot"].data.joint_pos[1] = 0
    env.common_step_counter += 1
    reverse.advance(env)
    assert reverse.success(env).tolist() == [True, False]
    assert mdp.state(env).hold_time[1] == 0


def test_standing_state_reads_are_idempotent_and_partial_reset_is_isolated():
    env = _standing_env()
    reverse.advance(env)
    held = mdp.state(env).hold_time.clone()
    for _ in range(3):
        reverse.update_state(env)
        reverse.success(env)
        for term in env.cfg.rewards.values():
            term.func(env, **term.params)
    torch.testing.assert_close(mdp.state(env).hold_time, held)
    mdp.reset_transformation(env, torch.tensor([1]))
    reverse.advance(env)
    torch.testing.assert_close(mdp.state(env).hold_time, torch.tensor([0.02, 0.0]))
    env.common_step_counter += 1
    reverse.advance(env)
    torch.testing.assert_close(mdp.state(env).hold_time, torch.tensor([0.04, 0.02]))


def test_grounded_snake_is_not_unsafe_merely_because_it_is_low():
    env = _standing_env()
    data = env.scene["robot"].data
    data.joint_pos[:] = 0
    data.body_com_pos_w[..., 2] = 0.05
    assert not mdp.unsafe(env).any()
    assert not reverse.final_condition(env).any()


@pytest.mark.parametrize("field", ["joint_pos", "body_com_pos_w", "body_com_lin_vel_w"])
def test_corrupt_reward_inputs_return_finite_rewards(field):
    env = _standing_env()
    data = getattr(env.scene["robot"].data, field)
    data[0] = float("nan")
    data[1] = float("inf")
    assert mdp.unsafe(env).all()
    for term in env.cfg.rewards.values():
        assert torch.isfinite(term.func(env, **term.params)).all()


@pytest.mark.parametrize("foot", [0, 1])
def test_inverted_sole_loses_alignment_credit_but_pose_reward_stays_independent(foot):
    env = _standing_env()
    data = env.scene["robot"].data
    data.site_quat_w[1, foot] = torch.tensor(
        [0.0, 1.0, 0.0, 0.0] if foot == 0 else [1.0, 0.0, 0.0, 0.0]
    )
    torch.testing.assert_close(
        reverse.sole_ground_alignment_reward(env), torch.tensor([1.0, -0.5])
    )
    torch.testing.assert_close(
        reverse.joint_standing_reward(env), torch.zeros(2)
    )


@pytest.mark.parametrize("foot", [0, 1])
def test_one_sideways_sole_loses_alignment_credit_but_keeps_ascent_reward(foot):
    env = _standing_env()
    data = env.scene["robot"].data
    data.body_link_lin_vel_w[:, 6, 2] = 0.03
    data.site_quat_w[1, foot] = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
    torch.testing.assert_close(
        reverse.sole_ground_alignment_reward(env),
        torch.tensor([1.0, -0.125 + 0.75 * math.exp(-9.0)]),
    )
    torch.testing.assert_close(
        reverse.base_ascent_velocity_reward(env), torch.ones(2)
    )


def test_sole_direction_reward_is_not_attenuated_by_foot_height():
    env = _standing_env()
    env.scene["robot"].data.site_pos_w[1, :, 2] = 0.06
    reward = reverse.sole_ground_alignment_reward(env)
    torch.testing.assert_close(reward, torch.ones(2))


@pytest.mark.parametrize(
    "x,y,distance", [(0.0, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.1, 0.0, 0.0),
                     (0.0, 0.03, 0.03), (0.2, 0.0, 0.1), (0.2, 0.03, math.hypot(0.1, 0.03)),
                     (0.4, 0.0, 0.3)]
)
def test_base_projection_penalty_uses_distance_to_support_segment(x, y, distance):
    env = _standing_env()
    data = env.scene["robot"].data
    data.site_pos_w[:, 0, :2] = torch.tensor([-0.1, 0.0])
    data.site_pos_w[:, 1, :2] = torch.tensor([0.1, 0.0])
    data.body_link_pos_w[:, 6, :2] = torch.tensor([x, y])
    torch.testing.assert_close(reverse.base_support_distance(env), torch.full((2,), distance))
    expected = min(distance / env.cfg.base_support_distance_std, 4.0)
    torch.testing.assert_close(reverse.base_support_distance_cost(env), torch.full((2,), expected))
    term = env.cfg.rewards["base_support_distance"]
    penalty = term.weight * term.func(env)
    torch.testing.assert_close(penalty, torch.full((2,), term.weight * expected))


def test_base_support_penalty_ignores_com_mass_and_height():
    env = _standing_env()
    data = env.scene["robot"].data
    data.site_pos_w[:, 0, :2] = torch.tensor([-0.1, 0.0])
    data.site_pos_w[:, 1, :2] = torch.tensor([0.1, 0.0])
    env.sim.model.body_mass[0] = 11.0
    data.body_com_pos_w[..., :2] = float("inf")
    data.body_link_pos_w[..., :2] = 10.0
    data.body_link_pos_w[:, 6, :2] = torch.tensor([0.03, 0.05])
    data.body_link_pos_w[:, 6, 2] = 5.0
    data.body_com_pos_w[..., 2] = 2.0
    data.site_pos_w[..., 2] = -1.0
    torch.testing.assert_close(reverse.base_support_distance(env), torch.full((2,), 0.05))
    torch.testing.assert_close(reverse.base_support_distance_cost(env), torch.ones(2))
    data.body_link_pos_w[1, 6, :2] = 0.0
    torch.testing.assert_close(reverse.base_support_distance(env), torch.tensor([0.05, 0.0]))
    torch.testing.assert_close(reverse.base_support_distance_cost(env), torch.tensor([1.0, 0.0]))


def test_base_support_penalty_is_invariant_to_translation_and_yaw():
    env = _standing_env()
    data = env.scene["robot"].data
    data.site_pos_w[:, 0, :2] = torch.tensor([-0.1, 0.0])
    data.site_pos_w[:, 1, :2] = torch.tensor([0.1, 0.0])
    data.body_link_pos_w[:, 6, :2] = torch.tensor([0.0, 0.03])
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]])
    offset = torch.tensor([10.0, -4.0])
    for positions in (data.site_pos_w, data.body_link_pos_w):
        positions[1, :, :2] = positions[1, :, :2] @ rotation.T + offset
    distance = reverse.base_support_distance(env)
    torch.testing.assert_close(distance, torch.full((2,), 0.03), atol=1e-6, rtol=1e-5)


def test_base_support_penalty_handles_coincident_sites_and_invalid_geometry():
    env = _standing_env()
    data = env.scene["robot"].data
    data.body_link_pos_w[:, 6, :2] = torch.tensor([0.03, 0.04])
    torch.testing.assert_close(reverse.base_support_distance(env), torch.full((2,), 0.05))
    data.site_pos_w[0, 0, 0] = float("nan")
    data.body_link_pos_w[1, 6, :2] = float("inf")
    torch.testing.assert_close(reverse.base_support_distance_cost(env), torch.full((2,), 4.0))


@pytest.mark.parametrize("std", [0.0, -0.05, float("nan"), float("inf")])
def test_base_support_penalty_rejects_invalid_distance_scale(std):
    env = _standing_env()
    env.cfg.base_support_distance_std = std
    with pytest.raises(ValueError, match="finite and positive"):
        reverse.base_support_distance_cost(env)


def test_balanced_sole_alignment_beats_one_perfect_foot_and_one_bad_foot():
    env = _standing_env()
    # 总角度相同：(30, 30) 与 (0, 60)。旧的均值计算偏向 (0, 60)。
    for row, angles in enumerate(((30.0, 30.0), (0.0, 60.0))):
        for foot, angle in enumerate(angles):
            half = math.radians(angle + 180.0 * foot) / 2
            env.scene["robot"].data.site_quat_w[row, foot] = torch.tensor(
                [math.cos(half), math.sin(half), 0.0, 0.0]
            )
    scores = reverse._sole_ground_alignment(env)
    assert scores.mean(-1)[0] < scores.mean(-1)[1]
    reward = reverse.sole_ground_alignment_reward(env)
    assert reward[0] > reward[1]
    assert scores[0, 0] > 0.3  # 在 30 度处仍提供较宽范围的反馈。


@pytest.mark.parametrize("bad_foot", [0, 1])
def test_sole_metrics_measure_each_foot_and_require_both_within_ten_degrees(bad_foot):
    env = _standing_env()
    half = math.radians(30.0 + 180.0 * bad_foot) / 2
    env.scene["robot"].data.site_quat_w[1, bad_foot] = torch.tensor(
        [math.cos(half), math.sin(half), 0.0, 0.0]
    )
    expected = torch.zeros(2, 2)
    expected[1, bad_foot] = 30.0
    torch.testing.assert_close(reverse.sole_angles_degrees(env), expected)
    for foot in (0, 1):
        metric = env.cfg.metrics[f"sole_{foot}_angle_deg"]
        assert metric.reduce == "mean"
        torch.testing.assert_close(metric.func(env, **metric.params), expected[:, foot])
    metric = env.cfg.metrics["both_soles_within_10deg"]
    assert metric.reduce == "mean"
    torch.testing.assert_close(metric.func(env), torch.tensor([1.0, 0.0]))
    env.scene["robot"].data.site_quat_w[0, bad_foot] = float("nan")
    assert reverse.sole_angles_degrees(env)[0, bad_foot] == 180.0
    assert reverse.both_soles_within_10deg(env)[0] == 0.0


@pytest.mark.parametrize("angle", [10.0, 14.0, 20.0])
def test_tilted_soles_cannot_collect_full_alignment_reward(angle):
    import math

    env = _standing_env()
    for foot in (0, 1):
        half = math.radians(angle + 180.0 * foot) / 2
        env.scene["robot"].data.site_quat_w[:, foot] = torch.tensor(
            [math.cos(half), math.sin(half), 0.0, 0.0]
        )
    assert (reverse.sole_ground_alignment_reward(env) < 0.9).all()
    assert not reverse.final_condition(env).any()


@pytest.mark.parametrize("height", [-0.01, 0.01, 0.05])
def test_sole_distance_cost_is_independent_of_orientation_and_penalizes_penetration(height):
    env = _standing_env()
    data = env.scene["robot"].data
    data.site_pos_w[1, :, 2] = height
    data.site_quat_w[1] = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
    cost = reverse.sole_height_cost(env)
    assert cost[0] == 0.0
    assert cost[1] > 0.0


@pytest.mark.parametrize("foot", [0, 1])
@pytest.mark.parametrize("height", [-0.03, 0.03, 0.30])
def test_sole_distance_penalty_counts_each_site_and_respects_ground_origin(foot, height):
    env = _standing_env()
    env.scene.env_origins[1, 2] = 2.0
    env.scene["robot"].data.site_pos_w[1, :, 2] = 2.0
    env.scene["robot"].data.site_pos_w[1, foot, 2] += height
    term = env.cfg.rewards["sole_height"]
    penalty = term.weight * term.func(env)
    expected = -0.5 * min(abs(height) / env.cfg.sole_alignment_height_std, 4.0)
    torch.testing.assert_close(penalty, torch.tensor([0.0, expected]))


def test_height_progress_has_signal_from_snake_and_is_independent_of_feet():
    env = _standing_env()
    data = env.scene["robot"].data
    data.body_link_pos_w[0, 6, 2] = reverse.SNAKE_ROOT_POS[2]
    data.body_link_pos_w[1, 6, 2] = reverse.SNAKE_ROOT_POS[2] + 0.01
    data.joint_pos[:] = 0.0
    data.site_quat_w[:] = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
    reward = reverse.base_height_progress_reward(env)
    assert reward[0].abs() < 1e-6
    assert reward[1] > 0.04


def test_pose_progress_is_independent_of_com_and_feet():
    env = _standing_env()
    data = env.scene["robot"].data
    data.body_com_pos_w[..., 2] = reverse.SNAKE_ROOT_POS[2]
    data.site_quat_w[:] = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
    data.joint_pos[0] = 0.0
    data.joint_pos[1] *= 0.1
    reward = reverse.joint_standing_reward(env)
    assert reward[0] < reward[1] < 0.0


@pytest.mark.parametrize("height", [0.0525, 0.0999, 0.1, 0.1001, 0.264])
def test_joint_reward_is_independent_of_base_origin_height(height):
    env = _standing_env()
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = height
    # 在任意基座高度，目标姿态和远离目标的姿态均保持各自的得分。
    target = env.scene["robot"].data.joint_pos[1].clone()
    env.scene["robot"].data.joint_pos[1] = -target
    far_score = -torch.nn.functional.smooth_l1_loss(-target, target, beta=env.cfg.joint_standing_error_scale)
    expected = torch.stack((torch.tensor(0.0), far_score))
    torch.testing.assert_close(reverse.joint_standing_reward(env), expected)


def test_joint_reward_is_independent_of_ground_origin_and_vertical_motion():
    env = _standing_env()
    data = env.scene["robot"].data
    env.scene.env_origins[1, 2] = 3.0
    data.body_link_pos_w[1, 6, 2] = 3.11
    torch.testing.assert_close(reverse.joint_standing_reward(env), torch.zeros(2))
    data.body_link_pos_w[1, 6, 2] = 3.09
    torch.testing.assert_close(reverse.joint_standing_reward(env), torch.zeros(2))


@pytest.mark.parametrize("height", [float("nan"), float("inf"), -float("inf")])
def test_joint_reward_does_not_read_base_height(height):
    env = _standing_env()
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = height
    torch.testing.assert_close(reverse.joint_standing_reward(env), torch.zeros(2))


def test_joint_reward_at_low_height_combines_with_live_curriculum_ramp():
    from mjlab.managers.reward_manager import RewardManager

    env = _standing_env()
    env.reward_manager = RewardManager(env.cfg.rewards, env)
    env._standing_curriculum_stage = 2
    env._standing_stage_start_step = 0
    env.common_step_counter = 1200
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = torch.tensor([0.09, 0.11])
    assert reverse.apply_standing_stage_weights(env) == 0.5
    env.scene["robot"].data.joint_pos[:, 0] += 1.0
    env.reward_manager.compute(env.step_dt)
    below = dict(env.reward_manager.get_active_iterable_terms(0))["joint_standing"][0]
    above = dict(env.reward_manager.get_active_iterable_terms(1))["joint_standing"][0]
    assert below == pytest.approx(above)
    assert above == pytest.approx(-0.75 / 6.0 * (
        env.cfg.first_stage_joint_weight
        + 0.5 * (env.cfg.standing_stage_joint_weight - env.cfg.first_stage_joint_weight)
    ))


@pytest.mark.parametrize("foot", [0, 1])
@pytest.mark.parametrize("violation", ["lift", "penetration", "tilt", "inverted", "invalid"])
def test_stage_two_positive_objectives_require_both_soles_down_and_grounded(foot, violation):
    env = _standing_env()
    env._standing_curriculum_stage = 2
    data = env.scene["robot"].data
    data.body_link_lin_vel_w[:, 6, 2] = 0.03
    for func in (reverse.base_height_progress_reward,
                 reverse.base_ascent_velocity_reward):
        torch.testing.assert_close(func(env), torch.ones(2))
    if violation in ("lift", "penetration"):
        data.site_pos_w[1, foot, 2] = 0.02 if violation == "lift" else -0.02
    elif violation == "invalid":
        data.site_quat_w[1, foot] = float("nan")
    else:
        angle = 30.0 if violation == "tilt" else 180.0
        half = math.radians(angle + 180.0 * foot) / 2
        data.site_quat_w[1, foot] = torch.tensor([math.cos(half), math.sin(half), 0.0, 0.0])
    assert reverse.sole_support_valid(env).tolist() == [True, False]
    for func in (reverse.base_height_progress_reward,
                 reverse.base_ascent_velocity_reward):
        torch.testing.assert_close(func(env), torch.tensor([1.0, 0.0]))
    data.joint_pos[1, 3] += 1.0
    torch.testing.assert_close(reverse.joint_standing_reward(env), torch.tensor([0.0, -0.75 / 6.0]))
    data.body_link_lin_vel_w[..., 2] = -0.03
    torch.testing.assert_close(reverse.base_ascent_velocity_reward(env), torch.full((2,), -3.0))
    data.body_link_pos_w[:, 6, 2] = 0.02
    assert (reverse.base_height_progress_reward(env) < 0).all()
    assert env.cfg.rewards["sole_height"].weight < 0
    assert env.cfg.rewards["sole_ground_alignment"].weight > 0


def test_support_loss_requires_prior_support_and_continuous_violation():
    from mjlab.managers.reward_manager import RewardManager

    env = _standing_env()
    env.cfg.standing_stage_ramp_steps = 0
    env.reward_manager = RewardManager(env.cfg.rewards, env)
    env._standing_curriculum_stage = 2
    data = env.scene["robot"].data
    data.site_pos_w[..., 2] = 0.05
    for step in range(10):
        env.common_step_counter = step
        reverse.advance(env)
        assert not reverse.foot_support_lost(env).any()
    # 两个环境均重新获得支撑，随后仅让环境 1 的一只脚失去支撑。
    data.site_pos_w[..., 2] = 0.0
    env.common_step_counter += 1
    reverse.advance(env)
    data.site_pos_w[1, 0, 2] = 0.02
    for step in range(1, 6):
        env.common_step_counter += 1
        reverse.advance(env)
        # 在同一控制步中重复读取不能累积额外时间。
        reverse.advance(env)
        assert reverse.foot_support_lost(env).tolist() == [False, step >= 5]
    data.site_pos_w[1, 0, 2] = 0.0
    env.common_step_counter += 1
    reverse.advance(env)
    assert not reverse.foot_support_lost(env).any()


def test_support_guard_reset_is_per_world_and_stage_one_does_not_enforce_it():
    from mjlab.managers.reward_manager import RewardManager

    env = _standing_env()
    env.cfg.standing_stage_ramp_steps = 0
    env.reward_manager = RewardManager(env.cfg.rewards, env)
    env._standing_curriculum_stage = 2
    reverse.advance(env)
    env.scene["robot"].data.site_pos_w[..., 2] = 0.02
    for step in range(1, 4):
        env.common_step_counter = step
        reverse.advance(env)
    reverse.reset_reverse_transition(env, torch.tensor([0]))
    task = reverse.support_protection_state(env)
    assert task.support_armed.tolist() == [False, True]
    assert task.support_violation_s[0] == 0.0
    assert task.support_violation_s[1] == pytest.approx(0.06)
    for step in range(4, 6):
        env.common_step_counter = step
        reverse.advance(env)
    assert reverse.foot_support_lost(env).tolist() == [False, True]
    env._standing_curriculum_stage = 1
    assert not reverse.foot_support_lost(env).any()
    torch.testing.assert_close(reverse.joint_standing_reward(env), torch.zeros(2))


def test_stage_one_support_hold_tracks_continuously_and_resets_per_world():
    env = _standing_env()
    task = reverse.support_protection_state(env)
    for step in range(25):
        env.common_step_counter = step
        reverse.advance(env)
        reverse.advance(env)  # 同一个时间步不得重复计数。
    torch.testing.assert_close(task.support_hold_s, torch.full((2,), 0.5))
    assert not task.support_armed.any()  # 第一阶段的失稳仍不能触发终止。
    env.scene["feet_ground_contact"].data.found[1, 0] = 0
    env.common_step_counter += 1
    reverse.advance(env)
    torch.testing.assert_close(task.support_hold_s, torch.tensor([0.52, 0.0]))
    reverse.reset_reverse_transition(env, torch.tensor([0]))
    torch.testing.assert_close(task.support_hold_s, torch.zeros(2))


def test_first_stage_joint_reward_is_active_in_low_snake_initial_pose():
    from mjlab.managers.reward_manager import RewardManager

    env = _standing_env()
    env.cfg = _recipe_snake_to_biped_env_cfg(play=False)
    env.scene["robot"].data.joint_pos[:] = torch.tensor(reverse.SNAKE_JOINT_POS)
    env.scene["robot"].data.body_link_pos_w[:, 6, 2] = reverse.SNAKE_ROOT_POS[2]
    score = reverse.joint_standing_reward(env)
    assert (score < 0.0).all()
    env.reward_manager = RewardManager(env.cfg.rewards, env)
    env.reward_manager.compute(env.step_dt)
    value = dict(env.reward_manager.get_active_iterable_terms(0))["joint_standing"][0]
    assert value == pytest.approx(env.cfg.first_stage_joint_weight * score[0].item())
    assert 0.0 < env.cfg.first_stage_joint_weight < env.cfg.standing_stage_joint_weight


@pytest.mark.parametrize("height", [0.0, 0.06, 0.3])
@pytest.mark.parametrize("angle,expected", [(0.0, 1.0), (90.0, -0.5 + math.exp(-9.0)), (180.0, -1.0)])
def test_each_sole_direction_has_signed_reward_without_height_escape(height, angle, expected):
    import math

    env = _standing_env()
    data = env.scene["robot"].data
    data.site_pos_w[..., 2] = height
    for foot in (0, 1):
        half = math.radians(angle + 180.0 * foot) / 2.0
        data.site_quat_w[:, foot] = torch.tensor([math.cos(half), math.sin(half), 0.0, 0.0])
    torch.testing.assert_close(
        reverse._sole_ground_alignment(env), torch.full((2, 2), expected)
    )


def test_bad_sole_alignment_does_not_disable_negative_speed_feedback():
    env = _standing_env()
    env.scene["robot"].data.site_quat_w[:] = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
    env.scene["robot"].data.body_link_lin_vel_w[0, 6, 2] = -0.03
    env.scene["robot"].data.body_link_lin_vel_w[1, 6, 2] = 0.12
    torch.testing.assert_close(
        reverse.base_ascent_velocity_reward(env), torch.tensor([-3.0, 1.0])
    )


def test_sole_alignment_is_invariant_to_world_yaw_and_ground_origin():
    env = _standing_env()
    data = env.scene["robot"].data
    offset = torch.tensor([10.0, -4.0, 3.0])
    env.scene.env_origins[1] = offset
    data.site_pos_w[1] += offset
    data.site_quat_w[1, 0] = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    data.site_quat_w[1, 1] = torch.tensor([0.0, 2**-0.5, 2**-0.5, 0.0])
    torch.testing.assert_close(reverse.sole_ground_alignment_reward(env), torch.ones(2))


@pytest.mark.parametrize("field", ["site_pos_w", "site_quat_w"])
def test_nonfinite_sole_geometry_gives_finite_alignment_and_ascent_rewards(field):
    env = _standing_env()
    data = env.scene["robot"].data
    getattr(data, field)[0] = float("nan")
    getattr(data, field)[1] = float("inf")
    data.body_link_lin_vel_w[:, 6, 2] = 0.03
    torch.testing.assert_close(
        reverse.sole_ground_alignment_reward(env),
        torch.ones(2) if field == "site_pos_w" else torch.zeros(2),
    )
    torch.testing.assert_close(reverse.base_ascent_velocity_reward(env), torch.ones(2))


def test_real_robot_standing_soles_align_better_than_grounded_snake_soles():
    import mujoco
    import numpy as np

    from zbot_rl_mjlab.robot import get_spec
    from zbot_rl_mjlab.tasks.transformation.poses import SNAKE_ROOT_POS, SNAKE_ROOT_QUAT

    model = get_spec().compile()
    native = mujoco.MjData(model)
    env = _standing_env()
    ids = [model.site(name).id for name in ("foot_0_sole", "foot_1_sole")]
    poses = (
        ((0.0, -0.06, 0.0), (1.0, 0.0, 0.0, 0.0), DEFAULT_JOINT_POS),
        (SNAKE_ROOT_POS, SNAKE_ROOT_QUAT, (0.0,) * 6),
    )
    for i, (pos, quat, joints) in enumerate(poses):
        native.qpos[:3], native.qpos[3:7], native.qpos[7:] = pos, quat, joints
        mujoco.mj_forward(model, native)
        if i == 0:
            assert native.xpos[model.body("base").id, 2] == pytest.approx(
                env.cfg.standing_base_height, abs=0.001
            )
            # 根据实际资产独立校准每个站点的符号，
            # 不依赖合成测试夹具及奖励中的符号常量。
            for site, sign in zip(ids, (1.0, -1.0)):
                assert native.site_xmat[site].reshape(3, 3)[2, 2] * sign > 0.999
        env.scene["robot"].data.site_pos_w[i] = torch.tensor(native.site_xpos[ids])
        for j, site in enumerate(ids):
            orientation = np.zeros(4)
            mujoco.mju_mat2Quat(orientation, native.site_xmat[site])
            env.scene["robot"].data.site_quat_w[i, j] = torch.tensor(orientation)
    score = reverse.sole_ground_alignment_reward(env)
    assert score[0] > 0.999
    assert score[1] < 0.05


def _rolled_env(angle_degrees, yaw_degrees=0.0):
    import math

    from mjlab.utils.lab_api.math import quat_mul

    env = _standing_env()
    # 边界回归测试使用固定限制，不受实时调参影响。
    env.cfg.base_z_roll_limit_deg = 90.0
    half = math.radians(angle_degrees) / 2
    yaw_half = math.radians(yaw_degrees) / 2
    ref = torch.tensor(SNAKE_ROOT_QUAT)
    local_twist = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
    world_yaw = torch.tensor([math.cos(yaw_half), 0.0, 0.0, math.sin(yaw_half)])
    root = quat_mul(world_yaw, quat_mul(ref, local_twist))
    env.scene["robot"].data.root_link_quat_w[:] = root
    env.scene["robot"].data.body_link_quat_w[:, 6] = quat_mul(
        world_yaw, quat_mul(torch.tensor([-0.3739189506, 0.3735469609, -0.6002646992, -0.6002745849]), local_twist)
    )
    return env


@pytest.mark.parametrize(
    "degrees", [-179.0, -90.001, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 90.001, 179.0]
)
def test_axial_roll_cost_and_strict_90_degree_reset_boundary(degrees):
    import math

    env = _rolled_env(degrees)
    torch.testing.assert_close(
        reverse.base_z_roll_angle(env),
        torch.full((2,), math.radians(degrees)),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        reverse.base_z_roll_cost(env),
        torch.full((2,), (degrees / 90.0) ** 2),
        atol=1e-5,
        rtol=1e-5,
    )
    assert reverse.base_z_roll_exceeded(env).tolist() == [abs(degrees) > 90.0] * 2


def test_axial_roll_is_detected_even_when_base_z_direction_does_not_change():
    from mjlab.utils.lab_api.math import quat_apply

    env = _rolled_env(120.0)
    initial = _rolled_env(0.0).scene["robot"].data.body_link_quat_w[:, 6]
    axis = torch.tensor([0.0, 0.0, 1.0]).expand(2, 3)
    torch.testing.assert_close(
        quat_apply(env.scene["robot"].data.body_link_quat_w[:, 6], axis),
        quat_apply(initial, axis),
    )
    assert reverse.base_z_roll_exceeded(env).all()


@pytest.mark.parametrize("yaw", [-90.0, 60.0, 90.0, 180.0])
@pytest.mark.parametrize("roll", [-120.0, -60.0, -30.0, 0.0, 30.0, 60.0, 120.0])
def test_heading_changes_do_not_change_axial_roll_or_reset_decision(yaw, roll):
    env = _rolled_env(roll, yaw)
    torch.testing.assert_close(
        reverse.base_z_roll_angle(env),
        reverse.base_z_roll_angle(_rolled_env(roll)),
        atol=1e-6,
        rtol=1e-6,
    )
    assert reverse.base_z_roll_exceeded(env).tolist() == [abs(roll) > 90.0] * 2


def test_quaternion_sign_changes_do_not_change_axial_roll():
    env = _rolled_env(30.0, 60.0)
    expected = reverse.base_z_roll_angle(env).clone()
    env.scene["robot"].data.root_link_quat_w[0] *= -1
    env.scene["robot"].data.body_link_quat_w[1, 6] *= -1
    torch.testing.assert_close(
        reverse.base_z_roll_angle(env), expected, atol=1e-6, rtol=1e-6
    )


def test_joint_induced_base_twist_is_measured_and_triggers_reset():
    import math

    from mjlab.utils.lab_api.math import quat_mul

    env = _rolled_env(0.0)
    joint_twist = torch.tensor([math.cos(0.8), 0.0, 0.0, math.sin(0.8)]).expand(2, 4)
    data = env.scene["robot"].data
    data.body_link_quat_w[:, 6] = quat_mul(data.body_link_quat_w[:, 6], joint_twist)
    torch.testing.assert_close(
        reverse.base_z_roll_angle(env), torch.full((2,), 1.6), atol=1e-6, rtol=0
    )
    assert reverse.base_z_roll_exceeded(env).all()


def test_base_roll_does_not_depend_on_root_orientation():
    env = _rolled_env(120.0)
    expected = reverse.base_z_roll_angle(env).clone()
    env.scene["robot"].data.root_link_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(reverse.base_z_roll_angle(env), expected)
    assert reverse.base_z_roll_exceeded(env).all()


def test_standing_roll_reference_follows_configured_joint_target():
    env = _rolled_env(0.0)
    torch.testing.assert_close(reverse.base_z_roll_angle(env), torch.zeros(2), atol=1e-6, rtol=0)
    env.cfg.standing_joint_pos = (0.0,) * 6
    assert abs(reverse.standing_roll_reference(env.cfg.standing_joint_pos)) < 1e-6
    assert reverse.base_z_roll_degrees(env)[0] > 60.0


def test_relative_roll_wraps_across_pi_and_invalid_orientation_still_resets():
    # 相对蛇形原始零位的 170 度，与 -170 度的参考角之间，
    # 角度差应为 -20 度，而非 340 度。
    from mjlab.utils.lab_api.math import quat_mul

    zero = torch.tensor([0.0, 0.0, -(0.5**0.5), -(0.5**0.5)])
    half = math.radians(170.0) / 2
    quat = quat_mul(zero, torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)]))
    angle = reverse.axial_roll_angle(quat, math.radians(-170.0))
    assert math.degrees(angle.item()) == pytest.approx(-20.0, abs=1e-4)
    invalid = torch.full((2, 4), float("nan"))
    torch.testing.assert_close(
        reverse.axial_roll_angle(invalid, math.pi), torch.full((2,), math.pi)
    )


@pytest.mark.parametrize("field", ["body_link_quat_w"])
def test_nonfinite_roll_geometry_is_finite_cost_and_resets(field):
    env = _standing_env()
    getattr(env.scene["robot"].data, field)[0] = float("nan")
    getattr(env.scene["robot"].data, field)[1] = float("inf")
    assert torch.isfinite(reverse.base_z_roll_cost(env)).all()
    assert reverse.base_z_roll_exceeded(env).all()


def test_vertical_base_axis_is_rejected_when_heading_is_undefined():
    env = _standing_env()
    env.scene["robot"].data.body_link_quat_w[:, 6] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert reverse.base_z_roll_exceeded(env).all()


def test_base_roll_limit_is_configurable_and_independent_per_environment():
    env = _rolled_env(31.0)
    env.cfg.base_z_roll_limit_deg = 30.0
    assert reverse.base_z_roll_exceeded(env).all()
    env.scene["robot"].data.root_link_quat_w[0] = torch.tensor(SNAKE_ROOT_QUAT)
    env.scene["robot"].data.body_link_quat_w[0, 6] = (
        _rolled_env(0.0).scene["robot"].data.body_link_quat_w[0, 6]
    )
    assert reverse.base_z_roll_exceeded(env).tolist() == [False, True]


def test_native_standing_is_roll_zero_and_snake_stays_inside_standing_corridor():
    import math

    import mujoco
    import numpy as np

    from zbot_rl_mjlab.robot import get_spec

    model = get_spec().compile()
    native = mujoco.MjData(model)
    base_id = model.body("base").id
    measured = []
    # 正向运动学的静态回归样本，不是策略使用的参考运动。
    for fraction in np.linspace(0.0, 1.0, 21):
        half = math.radians(-90 * (1 - fraction)) / 2
        root = (math.cos(half), math.sin(half), 0.0, 0.0)
        native.qpos[3:7] = root
        native.qpos[7:] = np.asarray(DEFAULT_JOINT_POS) * fraction
        mujoco.mj_forward(model, native)
        roll = reverse.axial_roll_angle(
            torch.tensor(native.xquat[base_id : base_id + 1], dtype=torch.float32),
            reverse.standing_roll_reference(DEFAULT_JOINT_POS),
        )
        measured.append(torch.rad2deg(roll).abs().item())
    assert 63.0 < measured[0] < 65.0
    assert measured[-1] < 0.01
    assert max(measured) < _recipe_snake_to_biped_env_cfg(play=True).base_z_roll_limit_deg


def test_native_standing_distinguishes_axial_roll_from_heading_changes():
    import math

    import mujoco
    import numpy as np

    from zbot_rl_mjlab.robot import get_spec

    model = get_spec().compile()
    native = mujoco.MjData(model)
    native.qpos[7:] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, native)
    base_id = model.body("base").id
    axis = native.xmat[base_id].reshape(3, 3)[:, 2].copy()
    baseline = reverse.axial_roll_angle(
        torch.tensor(native.xquat[base_id : base_id + 1], dtype=torch.float32)
    ).item()
    for yaw in (-90.0, 60.0, 90.0):
        half_yaw = math.radians(yaw) / 2
        yaw_quat = np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])
        for degrees in (-60.0, -30.0, 0.0, 30.0, 60.0):
            half_roll = math.radians(degrees) / 2
            roll_quat = np.r_[math.cos(half_roll), axis * math.sin(half_roll)]
            mujoco.mju_mulQuat(native.qpos[3:7], yaw_quat, roll_quat)
            mujoco.mj_forward(model, native)
            measured = reverse.axial_roll_angle(
                torch.tensor(native.xquat[base_id : base_id + 1], dtype=torch.float32),
            )
            assert abs(torch.rad2deg(measured).item() - math.degrees(baseline) - degrees) < 0.1


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_failure_cost_is_once_ten_points_independent_of_dt_and_excludes_timeouts(dt):
    from mjlab.managers.termination_manager import (
        TerminationManager,
        TerminationTermCfg,
    )

    env = _standing_env(3)
    env.step_dt = dt
    failure = torch.tensor([False, True, False])
    timeout = torch.tensor([False, True, True])
    manager = TerminationManager(
        {
            "base_z_roll": TerminationTermCfg(func=lambda env: failure),
            "unsafe": TerminationTermCfg(func=lambda env: failure),
            "time_out": TerminationTermCfg(func=lambda env: timeout, time_out=True),
        },
        env,
    )
    env.termination_manager = manager
    manager.compute()
    term = env.cfg.rewards["reset_failure"]
    actual = term.func(env) * term.weight * env.step_dt
    torch.testing.assert_close(actual, torch.tensor([0.0, -10.0, 0.0]))
    # 下一步清除终止掩码后，不应残留惩罚。
    failure[:] = False
    timeout[:] = False
    manager.compute()
    torch.testing.assert_close(term.func(env), torch.zeros(3))


def test_survival_uses_true_failure_mask_and_keeps_timeout_step_reward():
    from mjlab.managers.termination_manager import (
        TerminationManager,
        TerminationTermCfg,
    )

    env = _standing_env(4)
    failure = torch.tensor([False, True, False, True])
    timeout = torch.tensor([False, False, True, True])
    manager = TerminationManager(
        {"failure": TerminationTermCfg(func=lambda env: failure),
         "timeout": TerminationTermCfg(func=lambda env: timeout, time_out=True)},
        env,
    )
    env.termination_manager = manager
    manager.compute()
    assert manager.dones.tolist() == [False, True, True, True]
    torch.testing.assert_close(reverse.survival_reward(env), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    failure[:] = False
    timeout[:] = False
    manager.compute()
    assert reverse.survival_reward(env).eq(1).all()
    assert env.cfg.survival_weight == 0.0
    assert env.cfg.rewards["survival"].weight == 0.0


def test_reverse_success_still_requires_real_hold_despite_zero_observation_slot():
    env = _standing_env()
    task = mdp.state(env)
    term = env.cfg.observations["actor"].terms["hold_time"]
    for seconds, expected in ((0.0, False), (1.9, False), (2.0, True), (5.0, True)):
        task.hold_time[:] = seconds
        before = task.hold_time.clone()
        assert term.func(env).eq(0).all()
        assert torch.equal(task.hold_time, before)
        assert reverse.success(env).tolist() == [expected, expected]
