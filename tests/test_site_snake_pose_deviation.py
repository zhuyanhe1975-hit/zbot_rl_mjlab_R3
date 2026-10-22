"""Posture cost targets the snake reset, independently of biped defaults and actions."""

from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.experiments.site_snake import (
    site_snake_env_cfg as _recipe_site_snake_env_cfg,
)
from zbot_rl_mjlab.tasks.snake import site_omni as task


@pytest.mark.parametrize('play', [False, True])
def test_pose_cost_is_enabled_in_train_and_play(play):
    term = _recipe_site_snake_env_cfg(play).rewards['joint_pose_deviation']
    assert term.func is task.joint_pose_deviation_cost
    assert term.weight == -.02


def test_straight_snake_is_free_even_when_default_pose_is_biped():
    data = SimpleNamespace(joint_pos=torch.zeros(2, 6), default_joint_pos=torch.ones(2, 6))
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)})
    torch.testing.assert_close(task.joint_pose_deviation_cost(env), torch.zeros(2))


def test_cost_depends_on_actual_joint_bending_not_actions_or_world_yaw():
    position = torch.tensor([[.3]*6, [-.3]*6, [.6]*6, [0.]*6])
    data = SimpleNamespace(joint_pos=position, default_joint_pos=torch.ones_like(position),
                           root_link_quat_w=torch.randn(4, 4))
    env = SimpleNamespace(scene={'robot': SimpleNamespace(data=data)},
                          action_manager=SimpleNamespace(action=torch.randn(4, 6)))
    expected = torch.tensor([.09, .09, .36, 0.])
    torch.testing.assert_close(task.joint_pose_deviation_cost(env), expected)
    data.root_link_quat_w.normal_()
    env.action_manager.action.fill_(1.)
    torch.testing.assert_close(task.joint_pose_deviation_cost(env), expected)
