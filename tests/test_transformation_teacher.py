"""教师策略结构、基座与质心语义，以及特权价值网络的隔离性。"""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from zbot_rl_mjlab.experiments.transformation import (
    SnakeToBipedRobust,
    SnakeToBipedTeacher,
)
from zbot_rl_mjlab.tasks.transformation import teacher_observations as obs
from zbot_rl_mjlab.tasks.transformation.robust import RobustSnakeToBipedEnvCfg


def assemble(group, env):
    columns = []
    for term in group.terms.values():
        value = term.func(env, **term.params)
        if term.clip is not None:
            value = value.clamp(*term.clip)
        if term.scale is not None:
            value = value * term.scale
        columns.append(value)
    return torch.cat(columns, dim=-1)


@pytest.fixture
def teacher_env():
    class Scene(dict):
        pass

    cfg = SnakeToBipedTeacher().build_env()
    assert isinstance(cfg, RobustSnakeToBipedEnvCfg)
    origins = torch.tensor([[10.0, 20.0, 1.0], [30.0, 40.0, 2.0]])
    position = origins[:, None, :].expand(2, 3, 3).clone()
    position[:, 1, 2] += torch.tensor([0.2, 0.264])
    quaternion = torch.tensor([1.0, 0, 0, 0]).expand(2, 3, 4).clone()
    quaternion[1, 1] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    linear = torch.zeros(2, 3, 3)
    linear[:, 1] = torch.tensor([0.3, 0.4, 0.15])
    angular = torch.zeros(2, 3, 3)
    angular[:, 1, 0] = 1
    sites = origins[:, None, :].expand(2, 2, 3).clone()
    sites[:, :, 2] += torch.tensor([0.02, 0.03])
    site_quats = torch.tensor([[1.0, 0, 0, 0], [0.0, 1, 0, 0]]).expand(2, 2, 4).clone()
    data = SimpleNamespace(joint_pos=torch.tensor(cfg.standing_joint_pos).expand(2, 6).clone(), joint_vel=torch.full((2, 6), 0.3), body_link_pos_w=position, body_link_quat_w=quaternion, body_link_lin_vel_w=linear, body_link_ang_vel_w=angular, body_com_pos_w=position + 0.1, body_external_wrench=torch.zeros(2, 3, 6), actuator_force=torch.ones(2, 6), site_pos_w=sites, site_quat_w=site_quats)
    robot = SimpleNamespace(data=data, find_bodies=lambda name: ([1], [name]), find_sites=lambda names, preserve_order=True: ([0, 1], names), indexing=SimpleNamespace(body_ids=torch.arange(3)))
    scene = Scene(robot=robot, feet_ground_contact=SimpleNamespace(data=SimpleNamespace(found=torch.tensor([[1, 0], [0, 1]]))), base_ground_contact=SimpleNamespace(data=SimpleNamespace(found=torch.ones(2, 1))))
    scene.env_origins = origins
    action = SimpleNamespace(commanded_target=torch.zeros(2, 6), raw_action=torch.full((2, 3), 0.25), commanded_target_velocity=torch.full((2, 6), 0.5), rate_cfg=cfg.actions["joint_pos"])
    return SimpleNamespace(cfg=cfg, num_envs=2, device="cpu", common_step_counter=0, episode_length_buf=torch.zeros(2, dtype=torch.long), max_episode_length=800, scene=scene, action_manager=SimpleNamespace(get_term=lambda name: action), sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(2, 3))))


@pytest.mark.parametrize("play", [False, True])
def test_teacher_is_separate_from_legacy_robust_and_preserves_objective(play):
    cfg = load_env_cfg(obs.TEACHER_TASK_ID, play=play)
    original = SnakeToBipedRobust().build_env(play=play)
    assert isinstance(original, RobustSnakeToBipedEnvCfg)
    assert cfg.pretrained_checkpoint is None
    assert cfg.rewards == original.rewards and cfg.terminations == original.terminations
    assert cfg.initial_roll_range_deg == original.initial_roll_range_deg
    assert cfg.actions["joint_pos"].max_target_speed == 2 * math.pi
    assert original.actions["joint_pos"].max_target_speed == 2 * math.pi
    assert list(cfg.observations["actor"].terms) == ["teacher_state"]
    assert list(cfg.observations["critic"].terms) == ["teacher_critic_state"]
    assert set(cfg.metrics) == {"joint_error", "base_height", "base_vertical_velocity", "success", "joint_speed_peak"}
    assert all(sensor.fields == ("found",) for sensor in cfg.scene.sensors)
    assert {"com_state", "hold_time", "support_state"}.isdisjoint(cfg.observations["actor"].terms)
    assert {"external_wrench", "reset_memory", "com_relative_to_base"}.isdisjoint(cfg.observations["actor"].terms)
    agent = load_rl_cfg(obs.TEACHER_TASK_ID)
    assert not agent.resume and not agent.actor.obs_normalization and not agent.critic.obs_normalization
    assert agent.algorithm.learning_rate == 3e-4 and agent.algorithm.entropy_coef == 0.005


def test_teacher_dimensions_and_privileged_state_do_not_leak_into_actor(teacher_env):
    env = teacher_env
    actor = assemble(env.cfg.observations["actor"], env)
    critic = assemble(env.cfg.observations["critic"], env)
    assert actor.shape == (2, obs.TEACHER_ACTOR_DIM)
    assert critic.shape == (2, obs.TEACHER_CRITIC_DIM)
    assert torch.isfinite(actor).all() and torch.isfinite(critic).all()
    torch.testing.assert_close(critic[:, :36], actor)
    env.scene["robot"].data.body_com_pos_w += 0.4
    env.scene["robot"].data.body_external_wrench[:, 1] = 0.1
    torch.testing.assert_close(assemble(env.cfg.observations["actor"], env), actor)
    torch.testing.assert_close(assemble(env.cfg.observations["critic"], env), critic)
    from zbot_rl_mjlab.tasks.transformation.robust import fall_retry_state

    fall_retry_state(env).still_time[:] = 0.5
    torch.testing.assert_close(assemble(env.cfg.observations["actor"], env), actor)
    assert not torch.equal(assemble(env.cfg.observations["critic"], env), critic)


def test_base_height_velocity_and_gyro_are_in_the_correct_frames(teacher_env):
    env = teacher_env
    torch.testing.assert_close(obs.base_origin_height(env).flatten(), torch.tensor([0.2, 0.264]))
    torch.testing.assert_close(obs.base_world_vertical_velocity(env).flatten(), torch.tensor([0.15, 0.15]))
    torch.testing.assert_close(obs.base_linear_velocity(env), torch.tensor([[0.3, 0.4, 0.15], [0.4, -0.3, 0.15]]))
    torch.testing.assert_close(obs.base_angular_velocity(env), torch.tensor([[1.0, 0, 0], [0.0, -1, 0]]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(obs.sole_heights(env), torch.tensor([[0.02, 0.03], [0.02, 0.03]]))
    torch.testing.assert_close(obs.ground_contacts(env), torch.tensor([[1.0, 0, 1], [0.0, 1, 1]]))
    torch.testing.assert_close(obs.requested_target_speed_rms(env), torch.full((2,), math.pi / 2))
    torch.testing.assert_close(obs.target_speed_rms(env), torch.full((2,), 0.5))


def test_fused_actor_matches_reference_and_invalidates_on_reset(teacher_env):
    env = teacher_env
    reference = obs.teacher_reference_observation_groups()["actor"]
    before = obs.teacher_actor_state(env).clone()
    torch.testing.assert_close(before, assemble(reference, env))
    cached = obs.teacher_actor_state(env)
    assert cached.data_ptr() == obs.teacher_actor_state(env).data_ptr()
    env.scene["robot"].data.joint_pos[1, 0] += 0.2
    env._observation_generation = 1
    after = obs.teacher_actor_state(env)
    torch.testing.assert_close(after[0], before[0])
    assert not torch.equal(after[1], before[1])
    torch.testing.assert_close(after, assemble(reference, env))
    env.scene["robot"].data.joint_vel[:] = 20
    env.common_step_counter += 1
    torch.testing.assert_close(obs.teacher_actor_state(env), assemble(reference, env))
    assert (env.scene["robot"].data.joint_vel == 20).all()
