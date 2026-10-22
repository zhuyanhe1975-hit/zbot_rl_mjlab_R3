"""蛇形动作直接给绝对角度，不积累、不依赖dt、不保留上一步目标。"""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction

from zbot_rl_mjlab.experiments.snake import snake_env_cfg
from zbot_rl_mjlab.tasks.snake.actions import (
    DirectJointPositionAction,
    DirectJointPositionActionCfg,
)
from zbot_rl_mjlab.tasks.transformation.actions import JointTargetRateActionCfg
from zbot_rl_mjlab.tasks.transformation.env_cfg import transformation_scene_cfg

DIRECT_ACTION_SCALE = math.pi / 2


@pytest.fixture
def make_action(monkeypatch):
    def initialize(self, cfg, env):
        self._env = env
        self._target_ids = list(range(6))
        self._raw_actions = torch.zeros(2, 6)
        self._processed_actions = torch.zeros(2, 6)
        self._entity = SimpleNamespace(data=env.data)
        self._scale = cfg.scale
        self._offset = env.data.default_joint_pos.clone() if cfg.use_default_offset else cfg.offset

    monkeypatch.setattr(JointPositionAction, '__init__', initialize)

    def make(dt=.02, default=0., scale=None):
        data = SimpleNamespace(
            joint_pos_limits=torch.tensor([-6.28, 6.28]).expand(2, 6, 2).clone(),
            default_joint_pos=torch.full((2, 6), default),
        )
        cfg = DirectJointPositionActionCfg(entity_name='robot', actuator_names=('.*',), scale=math.pi / 2, use_default_offset=True)
        if scale is not None:
            cfg.scale = scale
        return cfg.build(SimpleNamespace(data=data, step_dt=dt))

    return make


@pytest.mark.parametrize('play', [False, True])
def test_snake_and_transformation_use_rate_control(play):
    cfg = snake_env_cfg(play=play).actions['joint_pos']
    assert isinstance(cfg, JointTargetRateActionCfg)
    assert cfg.target_limit == math.pi / 2
    assert cfg.max_target_speed == 2 * math.pi
    assert not cfg.use_default_offset
    assert isinstance(transformation_scene_cfg(play=play).actions['joint_pos'], JointTargetRateActionCfg)


@pytest.mark.parametrize('dt', [.005, .02, .1])
def test_action_maps_to_absolute_angle_without_integration(make_action, dt):
    action = make_action(dt)
    assert isinstance(action, DirectJointPositionAction)
    inputs = torch.tensor([[-1., -.5, 0., .25, .5, 1.], [1., .5, .25, 0., -.5, -1.]])
    action.process_actions(inputs)
    torch.testing.assert_close(action.commanded_target, inputs * DIRECT_ACTION_SCALE)
    for _ in range(30):
        action.process_actions(inputs)
    torch.testing.assert_close(action.commanded_target, inputs * DIRECT_ACTION_SCALE)
    action.process_actions(-inputs)
    torch.testing.assert_close(action.commanded_target, -inputs * DIRECT_ACTION_SCALE)
    action.process_actions(torch.zeros_like(inputs))
    torch.testing.assert_close(action.commanded_target, torch.zeros_like(inputs))


def test_clips_normalized_actions_and_respects_physical_joint_limits(make_action):
    action = make_action(scale=10.)
    action._entity.data.joint_pos_limits[:, 0] = torch.tensor([-.3, .4])
    action.process_actions(torch.full((2, 6), 2.))
    torch.testing.assert_close(action.raw_action, torch.ones(2, 6))
    torch.testing.assert_close(action.commanded_target[:, 0], torch.full((2,), .4))
    torch.testing.assert_close(action.commanded_target[:, 1:], torch.full((2, 5), math.pi))
    action.process_actions(torch.full((2, 6), -2.))
    torch.testing.assert_close(action.commanded_target[:, 0], torch.full((2,), -.3))
    torch.testing.assert_close(action.commanded_target[:, 1:], torch.full((2, 5), -math.pi))


def test_selected_reset_clears_action_and_target_without_touching_other_env(make_action):
    action = make_action()
    action.process_actions(torch.full((2, 6), .5))
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(action.commanded_target[0], torch.zeros(6))
    torch.testing.assert_close(action.raw_action[0], torch.zeros(6))
    torch.testing.assert_close(action.commanded_target[1], torch.full((6,), DIRECT_ACTION_SCALE / 2))
    action.reset()
    assert action.commanded_target.eq(0).all()
    assert action.raw_action.eq(0).all()


def test_nonfinite_inputs_produce_finite_bounded_targets(make_action):
    action = make_action()
    inputs = torch.tensor([[float('nan'), float('inf'), -float('inf'), 0., .5, -.5]]).expand(2, -1)
    action.process_actions(inputs)
    assert torch.isfinite(action.commanded_target).all()
    assert (action.commanded_target.abs() <= math.pi).all()
    assert action.commanded_target[:, 0].eq(0).all()


@pytest.mark.parametrize('limit', [0., -1., float('nan'), float('inf')])
def test_invalid_target_limits_are_rejected(limit):
    with pytest.raises(ValueError):
        DirectJointPositionActionCfg(entity_name='robot', actuator_names=('.*',), target_limit=limit)


def test_default_pose_offset_is_used_at_init_for_zero_action_and_selected_reset(make_action):
    action = make_action(default=.2)
    torch.testing.assert_close(action.commanded_target, torch.full((2, 6), .2))
    inputs = torch.full((2, 6), .5)
    action.process_actions(inputs)
    torch.testing.assert_close(action.commanded_target, .2 + inputs * DIRECT_ACTION_SCALE)
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(action.commanded_target[0], torch.full((6,), .2))
    torch.testing.assert_close(action.commanded_target[1], torch.full((6,), .2 + DIRECT_ACTION_SCALE / 2))
    action.process_actions(torch.zeros_like(inputs))
    torch.testing.assert_close(action.commanded_target, torch.full((2, 6), .2))
