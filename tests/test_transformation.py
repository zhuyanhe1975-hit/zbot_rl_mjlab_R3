"""独立关节零角度强化学习变形，无需参考运动或接触传感器。"""

from types import SimpleNamespace

import pytest
import torch

TASK = "Mjlab-Zbot-6dof-Transform-Biped-To-Snake"


def _goal_env(num_envs=2):
    class Scene(dict):
        pass

    positions = torch.zeros(num_envs, 12, 3)
    positions[..., 2] = 0.04
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(num_envs, 6),
            joint_vel=torch.zeros(num_envs, 6),
            body_com_pos_w=positions,
            body_com_lin_vel_w=torch.zeros(num_envs, 12, 3),
            body_link_pos_w=positions.clone(),
            body_link_lin_vel_w=torch.zeros(num_envs, 12, 3),
            body_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0])
            .expand(num_envs, 12, 4)
            .clone(),
        ),
        find_bodies=lambda names, preserve_order=True: ([6], names),
        indexing=SimpleNamespace(body_ids=torch.arange(12)),
    )
    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(num_envs, 3)
    action = SimpleNamespace(
        commanded_target=torch.zeros(num_envs, 6), raw_action=torch.zeros(num_envs, 6)
    )
    from zbot_rl_mjlab.tasks.transformation.env_cfg import transformation_env_cfg

    return SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        step_dt=0.02,
        common_step_counter=0,
        cfg=transformation_env_cfg(play=True),
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(12))),
        action_manager=SimpleNamespace(get_term=lambda name: action),
    )


def test_registered_task_uses_two_core_rewards_and_soft_symmetry_without_sensors():
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from zbot_rl_mjlab.tasks.transformation.actions import JointTargetRateActionCfg

    for play in (False, True):
        cfg = load_env_cfg(TASK, play=play)
        assert cfg.scene.sensors == ()
        assert isinstance(cfg.actions["joint_pos"], JointTargetRateActionCfg)
        actor = cfg.observations["actor"].terms
        assert set(actor) == {
            "joint_pos",
            "joint_vel",
            "commanded_target",
            "last_action",
            "com_state",
            "projected_gravity",
            "hold_time",
        }
        assert list(actor) == list(cfg.observations["critic"].terms)
        assert not any(
            "reference" in term.func.__name__ for term in cfg.rewards.values()
        )
        assert len(cfg.rewards) == 3
        assert (
            next(
                term.weight
                for term in cfg.rewards.values()
                if term.func.__name__ == "joint_symmetry_cost"
            )
            == -0.5
        )
        assert "success" not in cfg.terminations
        assert {term.func.__name__ for term in cfg.rewards.values()} == {
            "joint_zero_reward",
            "com_descent_velocity_reward",
            "joint_symmetry_cost",
        }
    assert load_rl_cfg(TASK).experiment_name == "zbot_transform_biped_to_snake"


def test_full_body_friction_is_local_to_transformation():
    from mjlab.tasks.registry import load_env_cfg

    from zbot_rl_mjlab.robot import get_spec

    old_before = get_spec()
    transformed = load_env_cfg(TASK).scene.entities["robot"].spec_fn()
    old_after = get_spec()
    old_by_name = {geom.name: geom for geom in old_before.geoms}
    bodies = [
        geom
        for geom in transformed.geoms
        if (geom.contype or geom.conaffinity)
        and not geom.name.startswith(("foot_0_", "foot_1_"))
    ]
    assert bodies
    for geom in bodies:
        assert geom.condim >= 3
        assert geom.friction[0] > 0
        assert old_by_name[geom.name].condim == 1
    assert [(g.name, g.condim, tuple(g.friction)) for g in old_before.geoms] == [
        (g.name, g.condim, tuple(g.friction)) for g in old_after.geoms
    ]


def test_walking_cli_addons_are_rejected_without_replacing_the_task():
    from mjlab.tasks import registry

    from zbot_rl_mjlab import cli

    before = dict(registry._REGISTRY)
    with pytest.raises(cli.CliError, match="does not support walking add-on switches"):
        cli.apply_cli([TASK, "--deployment"])
    assert registry._REGISTRY == before


def test_zero_angles_receive_more_reward_than_partially_folded_angles():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.scene["robot"].data.joint_pos[1] = 0.7
    reward = mdp.joint_zero_reward(env)
    assert reward[0] == 1.0
    assert 0 < reward[1] < reward[0]


def test_every_joint_must_be_near_zero_for_success():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.scene["robot"].data.joint_pos[1, 5] = env.cfg.joint_tolerance + 0.01
    assert mdp.final_condition(env).tolist() == [True, False]


def test_zero_angles_alone_do_not_accept_a_vertical_or_airborne_chain():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.scene["robot"].data.body_com_pos_w[1, 0, 2] = 0.4
    assert mdp.final_condition(env).tolist() == [True, False]


def test_goal_requires_low_joint_and_base_velocity():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.scene["robot"].data.joint_vel[0] = 1.0
    env.scene["robot"].data.body_link_lin_vel_w[1, 6, 0] = 0.03
    assert not mdp.final_condition(env).any()


def test_success_requires_two_continuous_seconds_and_resets_when_goal_is_lost():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    for step in range(99):
        env.common_step_counter = step
        mdp.advance(env)
    assert not mdp.success(env).any()
    env.common_step_counter = 99
    mdp.advance(env)
    assert mdp.success(env).all()
    env.scene["robot"].data.joint_vel[1] = 1.0
    env.common_step_counter += 1
    mdp.advance(env)
    assert mdp.success(env).tolist() == [True, False]
    assert mdp.state(env).hold_time[1] == 0


def test_reward_and_success_reads_do_not_advance_hold_time():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    mdp.advance(env)
    before = mdp.state(env).hold_time.clone()
    for _ in range(3):
        mdp.advance(env)
        for term in env.cfg.rewards.values():
            term.func(env, **term.params)
        mdp.success(env)
    torch.testing.assert_close(mdp.state(env).hold_time, before)


def test_reset_does_not_recount_the_same_step_or_reset_other_environment():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    mdp.advance(env)
    before = mdp.state(env).hold_time[0].clone()
    mdp.reset_transformation(env, torch.tensor([1]))
    mdp.advance(env)
    assert mdp.state(env).hold_time[1] == 0
    assert mdp.state(env).hold_time[0] == before
    env.common_step_counter += 1
    mdp.advance(env)
    assert mdp.state(env).hold_time[1] == pytest.approx(env.step_dt)


def test_fast_com_drop_is_unsafe_before_ground_impact():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.scene["robot"].data.body_com_lin_vel_w[1, :, 2] = -0.31
    assert mdp.unsafe(env).tolist() == [False, True]


def test_workspace_failure_uses_base_position_relative_to_environment_origin():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.scene.env_origins[0, 0] = 10.0
    env.scene["robot"].data.body_link_pos_w[0, 6, 0] = 10.0
    env.scene["robot"].data.body_link_pos_w[1, 6, 0] = 1.6
    assert mdp.unsafe(env).tolist() == [False, True]


def test_final_height_is_relative_to_each_environment_ground_origin():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    offset = torch.tensor([10.0, -4.0, 3.0])
    env.scene.env_origins[1] = offset
    env.scene["robot"].data.body_com_pos_w[1] += offset
    env.scene["robot"].data.body_link_pos_w[1] += offset
    assert mdp.final_condition(env).all()
    assert not mdp.unsafe(env).any()


@pytest.mark.parametrize(
    "field",
    [
        "joint_pos",
        "joint_vel",
        "body_com_pos_w",
        "body_com_lin_vel_w",
        "body_link_pos_w",
        "body_link_lin_vel_w",
        "body_link_quat_w",
    ],
)
def test_corrupt_terminal_states_fail_safely_and_all_rewards_remain_finite(field):
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    value = getattr(env.scene["robot"].data, field)
    value[0] = float("nan")
    value[1] = float("inf")
    assert mdp.unsafe(env).all()
    for name, term in env.cfg.rewards.items():
        assert torch.isfinite(term.func(env, **term.params)).all(), name


@pytest.mark.parametrize(
    "vertical_speed, expected",
    [
        (0.0, 0.0),
        (-0.03, 1.0),
        (0.03, -3.0),
        (-0.12, -4.0),
    ],
)
def test_com_descent_reward_favors_gentle_downward_motion(vertical_speed, expected):
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.scene["robot"].data.body_com_lin_vel_w[:, :, 2] = vertical_speed
    torch.testing.assert_close(
        mdp.com_descent_velocity_reward(env), torch.full((2,), expected)
    )


def test_com_reward_reads_mass_weighted_whole_body_velocity_instead_of_base():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.sim.model.body_mass[0] = 11.0
    env.scene["robot"].data.body_com_lin_vel_w[0, 0, 2] = -0.06
    env.scene["robot"].data.body_link_lin_vel_w[1, 6, 2] = -0.03
    torch.testing.assert_close(
        mdp.com_descent_velocity_reward(env), torch.tensor([1.0, 0.0])
    )


def test_com_state_uses_body_masses_and_environment_ground_origin():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.sim.model.body_mass[0] = 11.0
    env.scene["robot"].data.body_com_pos_w[:, 0, 2] = 0.08
    env.scene.env_origins[1, 2] = 3.0
    env.scene["robot"].data.body_com_pos_w[1, :, 2] += 3.0
    torch.testing.assert_close(
        mdp.com_state(env), torch.tensor([[0.06, 0.0], [0.06, 0.0]])
    )


def test_soft_symmetry_cost_uses_mirrored_order_and_opposite_sign():
    from zbot_rl_mjlab.tasks.transformation import mdp

    env = _goal_env()
    env.scene["robot"].data.joint_pos[:] = torch.tensor(
        [0.2, -0.4, 0.7, -0.7, 0.4, -0.2]
    )
    env.scene["robot"].data.joint_pos[1, 5] += 0.3
    torch.testing.assert_close(mdp.joint_symmetry_cost(env), torch.tensor([0.0, 0.03]))


def test_action_integrates_policy_velocity_instead_of_following_a_reference():
    from zbot_rl_mjlab.tasks.transformation.actions import bounded_target

    current = torch.tensor([[0.2, -0.4]])
    limits = torch.tensor([-3.0, 3.0]).expand(1, 2, 2)
    action = torch.tensor([[0.5, -0.5]])
    first = bounded_target(current, action, limits, 0.02, max_speed=1.0)
    second = bounded_target(first, action, limits, 0.02, max_speed=1.0)
    torch.testing.assert_close(first, torch.tensor([[0.21, -0.41]]))
    torch.testing.assert_close(second, torch.tensor([[0.22, -0.42]]))


def test_zero_action_preserves_the_current_commanded_target():
    from zbot_rl_mjlab.tasks.transformation.actions import bounded_target

    target = torch.tensor([[0.2, -0.4]])
    limits = torch.tensor([-3.0, 3.0]).expand(1, 2, 2)
    torch.testing.assert_close(
        bounded_target(target, torch.zeros_like(target), limits, 0.02), target
    )


def test_large_actions_cannot_bypass_target_speed_or_joint_limits():
    from zbot_rl_mjlab.tasks.transformation.actions import bounded_target

    target = torch.tensor([[0.99, -0.99, 0.0]])
    limits = torch.tensor([-1.0, 1.0]).expand(1, 3, 2)
    actual = bounded_target(
        target, torch.tensor([[100.0, -100.0, 100.0]]), limits, 0.02
    )
    torch.testing.assert_close(actual, torch.tensor([[1.0, -1.0, 0.02]]))


def _action_fixture(target_limit=None, max_speed=1.0):
    from zbot_rl_mjlab.tasks.transformation.actions import (
        JointTargetRateAction,
        JointTargetRateActionCfg,
    )

    action = JointTargetRateAction.__new__(JointTargetRateAction)
    action.rate_cfg = JointTargetRateActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        target_limit=target_limit,
        max_target_speed=max_speed,
    )
    action._raw_actions = torch.zeros(2, 6)
    action._processed_actions = torch.zeros(2, 6)
    action._target_ids = list(range(6))
    action._env = SimpleNamespace(step_dt=0.02)
    action._entity = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(2, 6),
            joint_pos_limits=torch.tensor([-6.28, 6.28]).expand(2, 6, 2).clone(),
        )
    )
    return action


def test_action_reset_uses_all_actual_joints_only_for_selected_environment():
    action = _action_fixture()
    action._raw_actions[:] = 1.0
    action._processed_actions[:] = 0.8
    actual = torch.tensor([0.2, -0.3, 0.4, 1.0, 1.1, 1.2])
    action._entity.data.joint_pos[1] = actual
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.commanded_target[0], torch.full((6,), 0.8))
    torch.testing.assert_close(action.commanded_target[1], actual)
    assert action._raw_actions[0].eq(1).all()
    assert action._raw_actions[1].eq(0).all()


def test_six_policy_actions_control_joints_independently_without_forced_symmetry():
    action = _action_fixture()
    command = torch.tensor([[1.0, -0.8, 0.6, 0.4, -0.2, 0.0]]).expand(2, 6)
    action.process_actions(command)
    torch.testing.assert_close(action.commanded_target, command * 0.02)


def test_each_joint_respects_its_own_model_limits():
    action = _action_fixture(max_speed=100.0)
    action._entity.data.joint_pos_limits[:, 5] = torch.tensor([-0.3, 0.6])
    action.process_actions(torch.ones(2, 6))
    torch.testing.assert_close(action.commanded_target[:, 0], torch.full((2,), 2.0))
    torch.testing.assert_close(action.commanded_target[:, 5], torch.full((2,), 0.6))


@pytest.mark.parametrize("target_limit", [None, 3.0])
def test_policy_can_cross_180_degrees_unless_a_stricter_limit_is_requested(
    target_limit,
):
    action = _action_fixture(target_limit)
    direction = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]).expand(2, 6)
    action._processed_actions[:] = direction * 2.99
    for _ in range(20):
        previous = action.commanded_target.clone()
        action.process_actions(direction)
        assert (action.commanded_target - previous).abs().max() <= 0.020001
    expected = 3.39 if target_limit is None else 3.0
    torch.testing.assert_close(action.commanded_target, direction * expected)
    for _ in range(200):
        action.process_actions(direction)
    expected = 6.28 if target_limit is None else 3.0
    torch.testing.assert_close(action.commanded_target, direction * expected)


def test_registered_action_uses_model_limits_by_default_and_asset_allows_past_pi():
    from mjlab.tasks.registry import load_env_cfg

    from zbot_rl_mjlab.robot import get_spec

    for play in (False, True):
        assert load_env_cfg(TASK, play=play).actions["joint_pos"].target_limit is None
    model = get_spec().compile()
    for name in (f"joint{i}" for i in range(1, 7)):
        low, high = model.jnt_range[model.joint(name).id]
        assert low < -torch.pi and high > torch.pi


def test_transform_exploration_settings_do_not_change_walking_defaults():
    from mjlab.tasks.registry import load_rl_cfg

    from zbot_rl_mjlab.rl import ppo_cfg

    before = ppo_cfg()
    transformed = load_rl_cfg(TASK)
    after = ppo_cfg()
    assert transformed.actor.distribution_cfg["init_std"] == 0.5
    assert transformed.actor.distribution_cfg["std_range"] == (0.1, 1.0)
    assert transformed.algorithm.entropy_coef == 0.005
    assert before.actor.distribution_cfg == after.actor.distribution_cfg
    assert before.algorithm.entropy_coef == after.algorithm.entropy_coef
