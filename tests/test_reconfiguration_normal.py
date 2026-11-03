"""Basic normal score must distinguish matching axes from face-to-face ports."""
import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from zbot_rl_mjlab.tasks.reconfiguration.basic import mdp


@pytest.mark.parametrize(('delta', 'tail_quat', 'expected'), [
    ((0., 0., .66), (1., 0., 0., 0.), 0.),  # Straight chain, backs facing.
    ((0., 0., -.02), (1., 0., 0., 0.), 1.),
    ((0., 0., -.6), (1., 0., 0., 0.), 1.),  # No distance attenuation.
    ((.02, 0., 0.), (1., 0., 0., 0.), .5),
    ((0., 0., -.02), (0., 1., 0., 0.), 0.),  # Opposite site Z axes.
    ((0., 0., 0.), (1., 0., 0., 0.), 1.),
    ((0., 0., 0.), (0., 1., 0., 0.), 0.),
    ((0., 0., -.02), (math.cos(.6), 0., 0., math.sin(.6)), 1.),
])
def test_normal_score_conventions_and_world_rotation(monkeypatch, delta, tail_quat, expected):
    g = {
        'q0': torch.tensor([[1., 0., 0., 0.]]),
        'q1': torch.tensor([tail_quat]),
        'delta': torch.tensor([delta]),
    }
    g['distance'] = g['delta'].norm(dim=-1)
    monkeypatch.setattr(mdp, 'state', lambda env: SimpleNamespace(geometry=lambda: g))
    torch.testing.assert_close(mdp.alignment_reward(None), torch.tensor([expected]))
    rotation = torch.tensor([[math.cos(.35), math.sin(.35), 0., 0.]])
    g['q0'] = quat_mul(rotation, g['q0'])
    g['q1'] = quat_mul(rotation, g['q1'])
    g['delta'] = quat_apply(rotation, g['delta'])
    torch.testing.assert_close(mdp.alignment_reward(None), torch.tensor([expected]))
