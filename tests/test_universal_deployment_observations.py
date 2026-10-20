"""综合部署学生的传感器历史观测与特权状态隔离契约。"""

import pytest
import torch
from test_universal_task import state_env as universal_state_env

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.deployment import configure_deployment
from zbot_rl_mjlab.tasks.universal.env_cfg import universal_env_cfg

DIMS = {
    "joint_pos": 6, "joint_vel": 6, "commanded_target": 6, "actions": 6,
    "projected_gravity": 3, "base_ang_vel": 3, "command": 4, "gait": 3,
}


def state_env():
    env, term = universal_state_env()
    env.scene["robot"].data.joint_pos_biased = env.scene["robot"].data.joint_pos.clone()
    return env, term


def observations(cfg, env):
    return {
        name: term.func(env, **term.params)
        for name, term in cfg.observations["actor"].terms.items()
    }


@pytest.mark.parametrize("play", [False, True])
def test_named_state_and_history_preserve_critic_and_base_factory(play):
    env, _ = state_env()
    cfg = universal_env_cfg(play=play)
    critic = cfg.observations["critic"]
    configure_deployment(cfg, play=play)
    assert cfg.observations["critic"] is critic
    assert not critic.enable_corruption and critic.history_length is None
    assert mdp.critic_state(env).shape == (2, 53)
    actor = cfg.observations["actor"]
    values = observations(cfg, env)
    assert {name: value.shape[1] for name, value in values.items()} == DIMS
    assert sum(DIMS.values()) == 37
    assert actor.history_length == 5 and actor.flatten_history_dim
    assert not actor.enable_corruption
    # 逐项无条件清理 NaN/Inf，替代管理器 sanitize 的逐项 .any() 同步。
    assert actor.nan_policy == "disabled"
    assert all(torch.isfinite(value).all() for value in values.values())
    assert 37 * actor.history_length == 185
    base = universal_env_cfg(play=play)
    assert list(base.observations["actor"].terms) == ["universal_state"]
    assert base.observations["actor"].history_length is None
    assert mdp.actor_state(env).shape == (2, 50)


def test_actor_terms_sanitize_non_finite_values():
    env, _ = state_env()
    cfg = universal_env_cfg()
    configure_deployment(cfg)
    env.scene["robot"].data.joint_pos_biased[0, 0] = float("nan")
    env.scene["robot"].data.joint_pos_biased[1, 1] = float("inf")
    value = cfg.observations["actor"].terms["joint_pos"].func(env)
    assert torch.isfinite(value).all()
    assert value[0, 0] == 0 and value[1, 1] == 0


def test_actor_and_critic_have_no_sensor_noise():
    cfg = universal_env_cfg()
    configure_deployment(cfg)
    terms = cfg.observations["actor"].terms
    assert all(term.noise is None for term in terms.values())
    assert all(term.noise is None for term in cfg.observations["critic"].terms.values())


def test_hidden_motion_height_and_contact_changes_do_not_leak_to_actor():
    env, term = state_env()
    cfg = universal_env_cfg()
    configure_deployment(cfg)
    before = observations(cfg, env)
    data = env.scene["robot"].data
    data.body_link_lin_vel_w[:] = 42
    data.body_link_pos_w[..., 2] += 1
    env.scene["feet_ground_contact"].data.found[:] = 1
    env.scene["base_ground_contact"].data.found[:] = 1
    term.initial_heading[:] = -term.initial_heading
    term.phase[:] = (term.phase + 1) % 4
    term.transition_mask[:] = ~term.transition_mask
    term.velocity_command[:] = torch.tensor([.2, -.1, .4])
    after = observations(cfg, env)
    for name in before:
        torch.testing.assert_close(after[name], before[name])


def test_gyro_is_body_frame_without_world_yaw_rate_input():
    env, _ = state_env()
    cfg = universal_env_cfg()
    configure_deployment(cfg)
    data = env.scene["robot"].data
    data.body_link_quat_w[:, 6] = torch.tensor([.5 ** .5, 0., .5 ** .5, 0.])
    data.body_link_ang_vel_w[:, 6] = torch.tensor([1., 0., 0.])
    value = observations(cfg, env)
    torch.testing.assert_close(value["base_ang_vel"], torch.tensor([[0., 0., 1.]]).expand(2, -1), atol=1e-6, rtol=0)
    assert "yaw_rate" not in value


def test_command_target_and_raw_action_are_current_unmodified_values():
    env, term = state_env()
    cfg = universal_env_cfg()
    configure_deployment(cfg)
    action = env.action_manager.get_term("joint_pos")
    action.commanded_target[:] = torch.arange(6) / 10
    action.raw_action[:] = torch.arange(6) / 7
    term.requested_velocity[:] = torch.tensor([.1, -.1, .3])
    term.velocity_command[:] = 0
    result = observations(cfg, env)
    torch.testing.assert_close(result["commanded_target"], action.commanded_target)
    torch.testing.assert_close(result["actions"], action.raw_action)
    torch.testing.assert_close(result["command"][:, 0], term.command[:, 0])
    torch.testing.assert_close(result["command"][:, 1:], term.requested_velocity)
    term.requested_velocity[:] = .2
    assert not torch.equal(observations(cfg, env)["command"], result["command"])


def test_encoder_bias_only_changes_actor_measured_joint_angles():
    env, _ = state_env()
    cfg = universal_env_cfg()
    configure_deployment(cfg)
    data = env.scene["robot"].data
    critic = mdp.critic_state(env).clone()
    data.joint_pos_biased += .015
    torch.testing.assert_close(observations(cfg, env)["joint_pos"], data.joint_pos + .015)
    torch.testing.assert_close(mdp.critic_state(env), critic)
