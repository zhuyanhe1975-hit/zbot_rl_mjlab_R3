"""Student PPO transformation costs do not freeze locomotion or forced recovery."""

from types import SimpleNamespace

import pytest
import torch
from test_teacher_archive_rewards import teacher_env

from zbot_rl_mjlab.tasks.universal import student_ppo_rewards as rewards


@pytest.fixture
def env_term():
    env, source = teacher_env()
    term = SimpleNamespace(
        transition_mask=torch.ones(2, dtype=torch.bool),
        current_task=torch.tensor([3, 1]), standing_mode=torch.tensor([True, False]),
        straightening_mask=torch.zeros(2, dtype=torch.bool), start_step=source.start_step,
    )
    env.command_manager.get_term = lambda name: term
    env.command_manager.get_command = lambda name: torch.tensor([[1.], [-1.]])
    env.sim = SimpleNamespace(model=SimpleNamespace(body_mass=torch.ones(2, 12)))
    data = env.scene['robot'].data
    data.body_link_pos_w[:, 6, 2] = torch.tensor([.25, .1])
    data.joint_pos[0] = data.joint_pos.new_tensor(env.cfg.standing_joint_pos)
    data.joint_pos[1] = 0
    return env, term


POLICY_COSTS = [rewards.transition_joint_overspeed, rewards.transition_settle_target,
                rewards.transition_settle_speed, rewards.transition_settle_action,
                rewards.transition_descent_speed]


@pytest.mark.parametrize('function', POLICY_COSTS + [rewards.transition_ground_impact])
def test_locomotion_has_no_transformation_cost(env_term, function):
    env, term = env_term
    env.scene['robot'].data.joint_vel[:] = 10
    term.transition_mask[:] = False
    term.current_task[:] = torch.tensor([0, 2])
    result = function(env)
    assert result.shape == (2,) and torch.isfinite(result).all()
    assert result.eq(0).all()
    # A stale transition flag cannot charge a locomotion task either.
    term.transition_mask[:] = True
    assert function(env).eq(0).all()


@pytest.mark.parametrize('function', POLICY_COSTS)
def test_policy_costs_exclude_executor_override(env_term, function):
    env, term = env_term
    term.straightening_mask[:] = True
    env.scene['robot'].data.joint_vel[:] = 10
    assert function(env).eq(0).all()


def test_joint_overspeed_uses_actual_speed_and_phase_limits(env_term):
    env, _ = env_term
    data = env.scene['robot'].data
    env.action_manager.get_term('joint_pos').commanded_target[:] = 100
    data.joint_vel[:] = 0
    assert rewards.transition_joint_overspeed(env).eq(0).all()
    data.joint_vel[:] = 3
    torch.testing.assert_close(rewards.transition_joint_overspeed(env), torch.tensor([0., .04]))
    data.joint_vel[:] = 1000
    assert rewards.transition_joint_overspeed(env).eq(4).all()


@pytest.mark.parametrize('function', [rewards.transition_settle_target, rewards.transition_settle_speed,
                                    rewards.transition_settle_action])
def test_settle_costs_only_at_end_of_transform(env_term, function):
    env, _ = env_term
    data = env.scene['robot'].data
    env.action_manager.get_term('joint_pos').commanded_target[:] = .1
    env.action_manager.get_term('joint_pos').raw_action[:] = .2
    data.joint_vel[:] = .2
    result = function(env)
    assert (result > 0).all() and torch.isfinite(result).all()
    data.body_link_pos_w[:, 6, 2] = torch.tensor([.1, .2])
    assert function(env).eq(0).all()
    data.body_link_pos_w[:, 6, 2] = torch.tensor([.25, .1])
    data.joint_pos[:] = 1
    assert function(env).eq(0).all()


def test_constant_residual_action_costs_despite_zero_delta(env_term):
    env, _ = env_term
    env.action_manager.action[:] = .2
    env.action_manager.prev_action[:] = .2
    env.action_manager.get_term('joint_pos').raw_action[:] = .2
    torch.testing.assert_close(rewards.transition_settle_action(env), torch.full((2,), .04))


def test_settle_target_uses_distinct_goal_for_each_direction(env_term):
    env, _ = env_term
    data = env.scene['robot'].data
    action = env.action_manager.get_term('joint_pos')
    action.commanded_target.copy_(data.joint_pos)
    assert rewards.transition_settle_target(env).eq(0).all()
    action.commanded_target[1] = .15
    torch.testing.assert_close(rewards.transition_settle_target(env), torch.tensor([0., 1.]))


def test_ascent_target_is_peak_and_latch_stops_credit(env_term):
    env, _ = env_term
    data = env.scene['robot'].data
    data.body_link_pos_w[:, 6, 2] = .1
    data.joint_pos[:] = 0
    data.body_link_lin_vel_w[:, 6, 2] = .1
    optimum = rewards.student_transition_ascent(env)
    data.body_link_lin_vel_w[:, 6, 2] = .15
    fast = rewards.student_transition_ascent(env)
    assert optimum[0] > fast[0] > 0
    assert optimum[1] == fast[1] == 0
    data.body_link_pos_w[:, 6, 2] = .25
    assert rewards.student_transition_ascent(env).eq(0).all()
    data.body_link_pos_w[:, 6, 2] = .1
    assert rewards.student_transition_ascent(env).eq(0).all()


def test_downward_speed_only_near_ground_and_descending(env_term):
    env, _ = env_term
    data = env.scene['robot'].data
    data.body_link_pos_w[:, 6, 2] = .1
    data.body_link_lin_vel_w[:, 6, 2] = -.2
    torch.testing.assert_close(rewards.transition_descent_speed(env), torch.tensor([0., 1.]))
    data.body_link_pos_w[:, 6, 2] = .2
    assert rewards.transition_descent_speed(env).eq(0).all()
    data.body_link_pos_w[:, 6, 2] = .1
    data.body_link_lin_vel_w[:, 6, 2] = .2
    assert rewards.transition_descent_speed(env).eq(0).all()


def test_ground_impact_applies_to_both_directions_and_forced_recovery(env_term):
    env, term = env_term
    term.straightening_mask[:] = True
    env.scene['transform_ground_contact'].data.force_history[..., 2] = 4 * 12 * 9.81
    torch.testing.assert_close(rewards.transition_ground_impact(env), torch.full((2,), 4.))


def test_nonfinite_residuals_are_bounded_instead_of_poisoning_rewards(env_term):
    env, _ = env_term
    data = env.scene['robot'].data
    data.joint_vel[:] = float('nan')
    env.action_manager.get_term('joint_pos').commanded_target[:] = float('inf')
    env.action_manager.get_term('joint_pos').raw_action[:] = float('nan')
    for function in POLICY_COSTS:
        assert torch.isfinite(function(env)).all()


@pytest.mark.parametrize(('function', 'parameter'), [
    (rewards.transition_joint_overspeed, 'ascent_limit'),
    (rewards.transition_joint_overspeed, 'descent_limit'),
    (rewards.transition_settle_target, 'angle_scale'),
    (rewards.transition_settle_speed, 'speed_scale'),
    (rewards.transition_settle_action, 'near_ground_height'),
    (rewards.transition_settle_action, 'near_goal_error'),
    (rewards.student_transition_ascent, 'target_speed'),
    (rewards.transition_descent_speed, 'max_down_speed'),
    (rewards.transition_descent_speed, 'approach_height'),
    (rewards.transition_ground_impact, 'max_body_weights'),
])
@pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan')])
def test_invalid_thresholds_rejected(env_term, function, parameter, value):
    with pytest.raises(ValueError, match='finite|positive|greater'):
        function(env_term[0], **{parameter: value})
