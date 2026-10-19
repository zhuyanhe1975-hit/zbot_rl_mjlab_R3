"""在不改变基础配方的前提下，为不同变形方向增加少量奖励。"""

from types import SimpleNamespace

import pytest
import torch
from test_transformation_commanded import environment

from zbot_rl_mjlab.experiments.bidirectional import SnakeBipedBidirectional
from zbot_rl_mjlab.experiments.transformation import SnakeToBipedTeacher
from zbot_rl_mjlab.tasks.transformation import commanded


@pytest.mark.parametrize("play", [False, True])
def test_two_increments_preserve_base_weights_and_teacher_task(play):
    cfg = SnakeBipedBidirectional().build_env(play=play)
    teacher = SnakeToBipedTeacher().build_env(play=play)
    increments = {"stand_finish_joint_speed", "descent_ground_impact"}
    assert cfg.rewards.keys() - teacher.rewards.keys() == increments
    for name in teacher.rewards:
        assert cfg.rewards[name].weight == teacher.rewards[name].weight
        assert cfg.rewards[name].params == teacher.rewards[name].params
    assert cfg.rewards["stand_finish_joint_speed"].weight == -0.02
    assert cfg.rewards["descent_ground_impact"].weight == -2
    assert cfg.actions == teacher.actions
    sensor = next(s for s in cfg.scene.sensors if s.name == "transform_ground_contact")
    assert sensor.primary.pattern == ".*" and sensor.secondary.pattern == "terrain"
    assert sensor.fields == ("force",) and sensor.reduce == "netforce"
    assert sensor.history_length == cfg.decimation


def test_finish_penalty_uses_actual_joint_speeds_and_only_current_high_stand_pose():
    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    data.joint_vel[:] = torch.tensor([1., -2., 3., -3., 2., -1.])
    data.body_link_pos_w[:, 6, 2] = torch.tensor([0.229, 0.23])
    expected = data.joint_vel.square().mean(-1)
    torch.testing.assert_close(commanded.stand_finish_joint_speed_cost(env), torch.tensor([0., expected[1]]))
    data.body_link_pos_w[:, 6, 2] = 0.264
    torch.testing.assert_close(commanded.stand_finish_joint_speed_cost(env), expected)
    term.set_target("snake")
    torch.testing.assert_close(commanded.stand_finish_joint_speed_cost(env), torch.zeros(2))
    term.set_target("stand")
    data.body_link_pos_w[:, 6, 2] = 0.1
    torch.testing.assert_close(commanded.stand_finish_joint_speed_cost(env), torch.zeros(2))


def test_finish_penalty_has_no_base_velocity_cost_and_respects_world_origin():
    env, term = environment()
    term._command.fill_(1)
    data = env.scene["robot"].data
    data.body_link_lin_vel_w.fill_(5)
    data.body_link_ang_vel_w.fill_(5)
    torch.testing.assert_close(commanded.stand_finish_joint_speed_cost(env), torch.zeros(2))
    data.joint_vel.fill_(2)
    before = commanded.stand_finish_joint_speed_cost(env)
    env.scene.env_origins[:, 2] += 10
    data.body_link_pos_w[..., 2] += 10
    torch.testing.assert_close(commanded.stand_finish_joint_speed_cost(env), before)


def test_impact_uses_peak_across_all_bodies_and_substeps_and_only_descent():
    env, term = environment()
    term._command.fill_(-1)
    history = torch.zeros(2, 12, env.cfg.decimation, 3)
    weight = 12 * 9.81
    history[:, 0, 0, 2] = weight
    env.scene["transform_ground_contact"] = SimpleNamespace(data=SimpleNamespace(force_history=history))
    torch.testing.assert_close(commanded.descent_ground_impact_cost(env), torch.zeros(2))
    history[1, 4, -1, 2] = 4 * weight
    torch.testing.assert_close(commanded.ground_force_peak(env), torch.tensor([weight, 4 * weight]))
    torch.testing.assert_close(commanded.descent_ground_impact_cost(env), torch.tensor([0., 4.]))
    term.set_target("stand")
    torch.testing.assert_close(commanded.descent_ground_impact_cost(env), torch.zeros(2))
    term.set_target("snake")
    env.sim.model.body_mass *= 2
    torch.testing.assert_close(commanded.descent_ground_impact_cost(env), torch.zeros(2))


def test_impact_normalization_uses_configured_gravity():
    env, term = environment()
    term._command.fill_(-1)
    env.cfg.sim.mujoco.gravity = (0., 0., -5.)
    history = torch.zeros(2, 12, env.cfg.decimation, 3)
    history[..., 2] = 180  # 在 g=5 时，十二个 1 kg 刚体总重的 3 倍。
    env.scene["transform_ground_contact"] = SimpleNamespace(data=SimpleNamespace(force_history=history))
    torch.testing.assert_close(commanded.descent_ground_impact_cost(env), torch.ones(2))


@pytest.mark.parametrize("threshold", [1., -1., float("nan"), float("inf")])
def test_invalid_impact_threshold_is_rejected(threshold):
    env, _term = environment()
    with pytest.raises(ValueError, match="max_body_weights"):
        commanded.descent_ground_impact_cost(env, max_body_weights=threshold)
