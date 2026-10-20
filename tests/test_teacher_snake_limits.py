"""全能教师仅在蛇形稳定运动阶段使用±60度目标角。"""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg
from test_transformation_action_bounds import (
    make_action as make_action,  # noqa: PLC0414
)

from zbot_rl_mjlab.tasks.universal.actions import (
    SnakeCrawlJointTargetRateAction,
    SnakeCrawlJointTargetRateActionCfg,
)


def action_fixture(make_action):
    base = make_action(target_limit=math.pi)
    term = SimpleNamespace(standing_mode=torch.tensor([False, True]), transition_mask=torch.tensor([False, False]))
    base._env.command_manager = SimpleNamespace(get_term=lambda _: term)
    cfg = SnakeCrawlJointTargetRateActionCfg(
        entity_name='robot', actuator_names=('.*',), target_limit=math.pi,
        max_target_speed=2 * math.pi,
    )
    return SnakeCrawlJointTargetRateAction(cfg, base._env), term


def test_teacher_train_and_play_register_snake_only_limit():
    for play in (False, True):
        cfg = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher', play=play)
        action = cfg.actions['joint_pos']
        assert isinstance(action, SnakeCrawlJointTargetRateActionCfg)
        assert action.snake_target_limit == pytest.approx(math.pi / 3)
        assert action.target_limit == math.pi
    other = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy')
    assert not isinstance(other.actions['joint_pos'], SnakeCrawlJointTargetRateActionCfg)


def test_crawl_clips_all_six_targets_without_windup_and_preserves_physics(make_action):
    action, _ = action_fixture(make_action)
    original = action._entity.data.joint_pos_limits.clone()
    sign = torch.tensor([[1., -1., 1., -1., 1., -1.]]).expand(2, -1)
    for _ in range(100):
        action.process_actions(sign)
    expected = sign * torch.tensor([[math.pi / 3], [math.pi]])
    torch.testing.assert_close(action.commanded_target, expected)
    action.process_actions(torch.zeros(2, 6))
    torch.testing.assert_close(action.commanded_target, expected)
    action.process_actions(-sign)
    torch.testing.assert_close(action.commanded_target, expected - sign * 2 * math.pi * .02)
    torch.testing.assert_close(action._entity.data.joint_pos_limits, original)


def test_both_transition_directions_keep_original_range_and_clamp_on_crawl_entry(make_action):
    action, term = action_fixture(make_action)
    term.transition_mask[:] = True
    for _ in range(100):
        action.process_actions(torch.ones(2, 6))
    torch.testing.assert_close(action.commanded_target, torch.full((2, 6), math.pi))
    term.transition_mask[0] = False
    action.process_actions(torch.zeros(2, 6))
    torch.testing.assert_close(action.commanded_target[0], torch.full((6,), math.pi / 3))
    torch.testing.assert_close(action.commanded_target[1], torch.full((6,), math.pi))
    term.standing_mode[0] = True
    for _ in range(100):
        action.process_actions(torch.ones(2, 6))
    torch.testing.assert_close(action.commanded_target, torch.full((2, 6), math.pi))


def test_reset_does_not_clip_new_standing_pose_using_previous_episode_snake_mode(make_action):
    action, _ = action_fixture(make_action)
    before = action.commanded_target[1].clone()
    action._entity.data.joint_pos[0] = torch.tensor([.312, .837, -2.02, 2.02, -.837, -.312])
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(action.commanded_target[0], action._entity.data.joint_pos[0])
    torch.testing.assert_close(action.commanded_target[1], before)
