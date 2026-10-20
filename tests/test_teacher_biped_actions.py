"""Archive-inspired residual walking without changing crawl/transform integration."""

import math
from types import SimpleNamespace
from typing import cast

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


def hybrid(make_action):
    base = make_action(initial=.5, target_limit=math.pi)
    term = SimpleNamespace(standing_mode=torch.tensor([True, False]), transition_mask=torch.zeros(2, dtype=torch.bool))
    base._env.command_manager = SimpleNamespace(get_term=lambda _: term)
    cfg = SnakeCrawlJointTargetRateActionCfg(
        entity_name='robot', actuator_names=('.*',), target_limit=math.pi,
        max_target_speed=2 * math.pi, biped_residual_scale=.25,
    )
    return SnakeCrawlJointTargetRateAction(cfg, base._env), term


def test_biped_targets_do_not_accumulate_and_zero_returns_to_nominal(make_action):
    action, _ = hybrid(make_action)
    for _ in range(100):
        action.process_actions(torch.ones(2, 6))
    torch.testing.assert_close(action.commanded_target[0], torch.full((6,), .75))
    torch.testing.assert_close(action.commanded_target[1], torch.full((6,), math.pi / 3))
    for _ in range(3):
        action.process_actions(torch.zeros(2, 6))
    torch.testing.assert_close(action.commanded_target[0], torch.full((6,), .5))
    torch.testing.assert_close(action.commanded_target[1], torch.full((6,), math.pi / 3))


def test_biped_entry_is_rate_limited_and_transform_still_integrates(make_action):
    action, term = hybrid(make_action)
    term.transition_mask[:] = True
    for _ in range(10):
        action.process_actions(torch.ones(2, 6))
    before = action.commanded_target.clone()
    term.transition_mask[0] = False
    action.process_actions(torch.zeros(2, 6))
    max_step = 2 * math.pi * .02
    torch.testing.assert_close(action.commanded_target[0], before[0] - max_step)
    torch.testing.assert_close(action.commanded_target[1], before[1])
    for _ in range(20):
        action.process_actions(torch.zeros(2, 6))
    torch.testing.assert_close(action.commanded_target[0], torch.full((6,), .5))


def test_biped_scale_is_teacher_only():
    teacher = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Teacher')
    action = teacher.actions['joint_pos']
    assert isinstance(action, SnakeCrawlJointTargetRateActionCfg)
    assert action.biped_residual_scale == .25
    assert action.transition_end_joint_radius == 1.
    student = load_env_cfg('Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy')
    assert not isinstance(student.actions['joint_pos'], SnakeCrawlJointTargetRateActionCfg)


@pytest.mark.parametrize('scale', [0., -1., float('inf'), float('nan')])
def test_invalid_biped_scale_is_rejected(scale):
    with pytest.raises(ValueError, match='biped_residual_scale'):
        SnakeCrawlJointTargetRateActionCfg(entity_name='robot', actuator_names=('.*',), biped_residual_scale=scale)


def test_end_joint_bounds_restrict_only_transform_and_preserve_other_joints(make_action):
    action, term = hybrid(make_action)
    cast(SnakeCrawlJointTargetRateActionCfg, action.rate_cfg).transition_end_joint_radius = 1.
    term.transition_mask[:] = True
    for _ in range(40):
        action.process_actions(torch.ones(2, 6))
    torch.testing.assert_close(action.commanded_target[:, (0, 5)], torch.full((2, 2), 1.5))
    torch.testing.assert_close(action.commanded_target[:, 1:5], torch.full((2, 4), math.pi))
    # Source walking/crawl test fixture defaults are .5; actual nominal end joints
    # are +.312/-.312, giving archive [-.688,1.312]/[-1.312,.688].
    term.transition_mask[1] = False
    action.process_actions(torch.zeros(2, 6))
    assert action.commanded_target[1].abs().max() <= math.pi / 3
