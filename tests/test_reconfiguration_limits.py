"""A persistent soft limit cost for actual angles and velocity-integrated targets."""
import math
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.tasks.reconfiguration.basic.mdp import limit_proximity_cost
from zbot_rl_mjlab.tasks.transformation.actions import JointTargetRateAction


def make_env(actual, target, *, limits=(-math.pi, math.pi), bound=2.):
    action = object.__new__(JointTargetRateAction)
    action._target_ids = slice(None)
    action.rate_cfg = SimpleNamespace(target_limit=bound, max_target_speed=math.pi)
    action._named_target_bounds = None
    action._entity = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.tensor(actual, dtype=torch.float32),
        joint_pos_limits=torch.tensor(limits).expand(len(actual), 6, 2).clone(),
    ))
    action._processed_actions = torch.tensor(target, dtype=torch.float32)
    action._raw_actions = torch.zeros_like(action._processed_actions)
    env = SimpleNamespace(step_dt=.02, action_manager=SimpleNamespace(get_term=lambda name: action))
    action._env = env
    return env, action


@pytest.mark.parametrize(('angle', 'expected'), [
    (0., 0.), (math.pi / 2, 0.), (1.7, 0.),
    (1.85, .25), (-1.85, .25), (2., 1.), (-2., 1.), (2.15, 2.25),
])
def test_smooth_buffer_and_uncapped_overshoot(angle, expected):
    env, _ = make_env([[angle] * 6], [[0.] * 6])
    torch.testing.assert_close(limit_proximity_cost(env), torch.tensor([expected]), atol=1e-6, rtol=1e-5)


def test_worst_actual_or_target_per_joint_then_mean_without_double_charge():
    env, action = make_env([[2., 0., 0., 0., 0., 0.]], [[0., -2., 0., 0., 0., 0.]])
    torch.testing.assert_close(limit_proximity_cost(env), torch.tensor([2/6]))
    action._processed_actions[0, 0] = 2.
    torch.testing.assert_close(limit_proximity_cost(env), torch.tensor([2/6]))
    # Recovering either joint improves the cost even if the other stays at a limit.
    action._entity.data.joint_pos[0, 0] = 0.
    action._processed_actions[0, 0] = 0.
    torch.testing.assert_close(limit_proximity_cost(env), torch.tensor([1/6]))


def test_uses_actual_effective_limits_including_asymmetric_physical_bounds():
    env, _ = make_env([[-.85] * 6], [[0.] * 6], limits=(-1., 2.5))
    torch.testing.assert_close(limit_proximity_cost(env), torch.tensor([.25]))
    env, _ = make_env([[1.35] * 6], [[0.] * 6], bound=1.5)
    torch.testing.assert_close(limit_proximity_cost(env), torch.tensor([.25]))


def test_persistent_cost_has_no_exit_bonus_or_memory_and_no_hidden_integral():
    env, action = make_env([[0.] * 6], [[2.] * 6])
    ones = torch.ones(1, 6)
    for _ in range(4):
        action.process_actions(ones)
        torch.testing.assert_close(limit_proximity_cost(env), torch.ones(1))
    action.process_actions(-ones)
    assert 0 < limit_proximity_cost(env).item() < 1
    for _ in range(4):
        action.process_actions(-ones)
    assert limit_proximity_cost(env).item() == 0.
    for _ in range(5):
        action.process_actions(ones)
    torch.testing.assert_close(limit_proximity_cost(env), torch.ones(1))


@pytest.mark.parametrize('margin', [0., -.1, math.inf, math.nan])
def test_invalid_buffer_is_rejected(margin):
    with pytest.raises(ValueError, match='margin'):
        limit_proximity_cost(None, margin=margin)
