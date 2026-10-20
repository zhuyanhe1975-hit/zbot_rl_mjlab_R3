"""大角度实际姿态和目标均需付费，保留180度控制及现有复位。"""

import math

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager
from test_snake_task import state_env

from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


@pytest.mark.parametrize('degrees,expected', [(0., 0.), (90., 0.), (120., 0.),
                                             (135., .0625), (150., .25),
                                             (179., (59 / 60) ** 2), (180., 1.), (210., 1.)])
@pytest.mark.parametrize('source', ['actual', 'target'])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_fold_cost_ramps_symmetrically_and_does_not_dilute_single_joint(degrees, expected, source, sign):
    env = state_env()
    values = (env.scene['robot'].data.joint_pos if source == 'actual'
              else env.action_manager.get_term('joint_pos').commanded_target)
    values[0, 2] = sign * math.radians(degrees)
    torch.testing.assert_close(mdp.joint_fold_cost(env), torch.tensor([expected, 0.]), atol=1e-6, rtol=1e-5)


def test_fold_cost_takes_worse_of_actual_and_target_without_double_charge():
    env = state_env()
    actual = env.scene['robot'].data.joint_pos
    target = env.action_manager.get_term('joint_pos').commanded_target
    actual[0, 0], target[0, 0] = math.pi, -math.pi
    actual[1, 0], target[1, 1] = math.radians(135), math.radians(150)
    torch.testing.assert_close(mdp.joint_fold_cost(env), torch.tensor([1., .25]))
    actual[0], target[0] = math.pi, -math.pi
    torch.testing.assert_close(mdp.joint_fold_cost(env), torch.tensor([1., .25]))


def test_inward_target_does_not_hide_still_folded_actual_joint_and_cost_falls_when_recovered():
    env = state_env()
    actual = env.scene['robot'].data.joint_pos
    target = env.action_manager.get_term('joint_pos').commanded_target
    actual[:, 0], target[:, 0] = math.pi, math.pi
    previous = mdp.joint_fold_cost(env).clone()
    target[:, 0] = 0.
    torch.testing.assert_close(mdp.joint_fold_cost(env), previous)
    actual[:, 0] = math.radians(150)
    assert (mdp.joint_fold_cost(env) < previous).all()
    actual[:, 0] = math.radians(120)
    torch.testing.assert_close(mdp.joint_fold_cost(env), torch.zeros(2), atol=1e-10, rtol=0.)


def test_cost_is_relative_to_initial_joint_pose():
    env = state_env()
    data = env.scene['robot'].data
    data.default_joint_pos[:] = .1
    data.joint_pos[:] = .1
    env.action_manager.get_term('joint_pos').commanded_target[:] = .1
    torch.testing.assert_close(mdp.joint_fold_cost(env), torch.zeros(2))


@pytest.mark.parametrize('dt', [.01, .02, .1])
def test_reward_manager_applies_weight_and_dt_once(dt):
    env = state_env()
    env.max_episode_length_s = 30.
    env.scene['robot'].data.joint_pos[:, 0] = math.radians(150)
    manager = RewardManager({'joint_fold_cost': env.cfg.rewards['joint_fold_cost']}, env)
    torch.testing.assert_close(manager.compute(dt), torch.full((2,), -.25 * dt))


@pytest.mark.parametrize('soft,full', [(-1., math.pi), (math.pi, math.pi),
                                      (math.pi, 0.), (float('nan'), math.pi),
                                      (0., float('inf'))])
def test_invalid_fold_thresholds_rejected(soft, full):
    with pytest.raises(ValueError, match='折叠'):
        mdp.joint_fold_cost(state_env(), soft_angle=soft, full_angle=full)


@pytest.mark.parametrize('play', [False, True])
def test_train_and_play_keep_range_observations_and_terminations(play):
    cfg = snake_env_cfg(play=play)
    assert cfg.actions['joint_pos'].target_limit == math.pi / 3
    term = cfg.rewards['joint_fold_cost']
    assert term.func is mdp.joint_fold_cost
    assert term.weight == -1.
    assert term.params == {'soft_angle': 2 * math.pi / 3, 'full_angle': math.pi}
    assert set(cfg.terminations) == {'unsafe', 'head_tilt', 'head_yaw', 'body_yaw', 'self_collision', 'time_out'}
    env = state_env()
    assert mdp.actor_state(env).shape == (2, 51)
    assert mdp.critic_state(env).shape == (2, 66)
