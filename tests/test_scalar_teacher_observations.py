"""Compact teacher observations preserve latched state and legacy playback."""

from types import SimpleNamespace
from typing import Literal, cast

import pytest
import torch

from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.tasks.universal.observation_layout import (
    teacher_motion_fields,
    teacher_task_ids,
)
from zbot_rl_mjlab.tasks.universal.privileged import (
    PrivilegedOmniCommandCfg,
    privileged_state,
)


def teacher_env():
    from test_universal_task import state_env

    env, _ = state_env()
    env._omni_initial_standing = torch.tensor([True, False])
    env._omni_play_standing = env._omni_initial_standing.clone()
    command = PrivilegedOmniCommandCfg(
        training=False, mode='source', resampling_time_range=(6., 12.), task_observation_encoding='scalar',
    ).build(env)
    env.command_manager.get_term = lambda _: command
    command.reset(torch.arange(2))
    return env, command


def test_scalar_state_preserves_all_other_fields_and_legacy_exactly():
    env, command = teacher_env()
    raw = torch.cat((mdp.critic_state(env), mdp.tracking_velocity(env), command.requested_velocity), -1)
    scalar = privileged_state(env)
    assert scalar.shape == (2, 56)
    torch.testing.assert_close(scalar[:, :38], raw[:, :38])
    torch.testing.assert_close(scalar[:, 40:], raw[:, 43:])
    torch.testing.assert_close(scalar[:, 38], command.current_morphology.float())
    torch.testing.assert_close(scalar[:, 39], command.current_task.float())
    command.teacher_cfg.task_observation_encoding = 'one_hot'
    torch.testing.assert_close(privileged_state(env), raw)


def test_scalar_state_encodes_selected_transition_and_completion():
    env, command = teacher_env()
    command.set_target('snake', torch.tensor([0]))
    observed = privileged_state(env)
    assert observed[:, 38].tolist() == [0., -1.]
    assert teacher_task_ids(observed).tolist() == [1, 2]
    command.set_target('stand', torch.tensor([1]))
    observed = privileged_state(env)
    assert observed[:, 38].tolist() == [0., 0.]
    assert teacher_task_ids(observed).tolist() == [1, 3]
    command.task_fsm.update_completion(torch.tensor([True, False]))
    observed = privileged_state(env)
    assert observed[:, 38].tolist() == [-1., 0.]
    assert teacher_task_ids(observed).tolist() == [2, 3]
    command.task_fsm.update_completion(torch.tensor([False, True]))
    observed = privileged_state(env)
    assert observed[:, 38].tolist() == [-1., 1.]
    assert teacher_task_ids(observed).tolist() == [2, 0]


@pytest.mark.parametrize('dimension', [56, 59])
def test_motion_layout_supports_arbitrary_leading_dimensions(dimension):
    observed = torch.arange(2 * 3 * dimension).reshape(2, 3, dimension).float()
    command, actual, pending = teacher_motion_fields(observed)
    start = 40 if dimension == 56 else 43
    torch.testing.assert_close(command, observed[..., start:start + 3])
    torch.testing.assert_close(actual, observed[..., -6:-3])
    torch.testing.assert_close(pending, observed[..., -3:])
    assert teacher_task_ids(observed).shape == (2, 3)
    assert teacher_task_ids(observed).dtype == torch.long


@pytest.mark.parametrize('shape', [(), (2, 55), (2, 3, 58)])
@pytest.mark.parametrize('decode', [teacher_task_ids, teacher_motion_fields])
def test_bad_layout_fails_clearly(shape, decode):
    with pytest.raises(ValueError, match='56 or 59'):
        decode(torch.zeros(shape))


def test_scalar_state_sanitizes_nonfinite_values(monkeypatch):
    env, _ = teacher_env()
    monkeypatch.setattr(mdp, 'tracking_velocity', lambda _: torch.full((2, 3), float('nan')))
    assert torch.isfinite(privileged_state(env)).all()


def test_unknown_encoding_is_rejected():
    cfg = PrivilegedOmniCommandCfg(resampling_time_range=(6., 12.))
    cfg.task_observation_encoding = cast(Literal['scalar', 'one_hot'], 'unknown')
    with pytest.raises(ValueError, match='task_observation_encoding'):
        cfg.build(SimpleNamespace())
