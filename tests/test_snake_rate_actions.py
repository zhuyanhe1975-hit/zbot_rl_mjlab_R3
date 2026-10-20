"""恢复速度积分后，零动作保持目标、顶限不累积、反向立即离开。"""

import math

import torch
from test_transformation_action_bounds import (
    make_action as make_action,  # noqa: PLC0414
)

from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


def test_snake_rate_mapping_integrates_holds_and_reverses_without_windup(make_action):
    cfg = snake_env_cfg(play=True).actions['joint_pos']
    action = make_action(target_limit=cfg.target_limit)
    action.rate_cfg.max_target_speed = cfg.max_target_speed
    direction = torch.tensor([[1., -1., 1., -1., 1., -1.]]).expand(2, -1)
    action.process_actions(direction)
    torch.testing.assert_close(action.commanded_target, direction * 2 * math.pi * .02)
    for _ in range(99):
        action.process_actions(direction)
    torch.testing.assert_close(action.commanded_target, direction * cfg.target_limit)
    action.process_actions(torch.zeros_like(direction))
    torch.testing.assert_close(action.commanded_target, direction * cfg.target_limit)
    action.process_actions(-direction)
    torch.testing.assert_close(action.commanded_target, direction * (cfg.target_limit - 2 * math.pi * .02))


def test_snake_rate_reset_keeps_nonreset_env_and_uses_actual_joint_position(make_action):
    cfg = snake_env_cfg(play=True).actions['joint_pos']
    action = make_action(target_limit=cfg.target_limit)
    action.rate_cfg.max_target_speed = cfg.max_target_speed
    action.process_actions(torch.ones(2, 6))
    previous = action.commanded_target[0].clone()
    action._entity.data.joint_pos[1] = 1.2
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.commanded_target[0], previous)
    torch.testing.assert_close(action.commanded_target[1], torch.full((6,), min(1.2, cfg.target_limit)))
