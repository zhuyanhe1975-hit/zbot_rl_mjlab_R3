"""History ordering, task isolation and exported student action contracts."""

import io
from typing import cast

import pytest
import torch
from mjlab.utils.buffers import CircularBuffer
from tensordict import TensorDict
from torch import nn

from zbot_rl_mjlab.gaussian_action_models import GaussianTaskDistribution, expand_mirror
from zbot_rl_mjlab.student_models import (
    HistoryGaussianStudentMLP,
    HistoryTaskEncoding,
    HistoryTaskNormalization,
    HistoryTaskRoutedNetwork,
)


def observations():
    frames = torch.randn(8, 5, 56)
    frames[:, :, 39] = torch.tensor([0., 1., 2., 3., 0.]).expand(8, -1)
    frames[:, -1, 39] = torch.tensor([0., 1., 2., 3.25] * 2)
    frames[:, :, 38] = 1
    frames[:, :, 46] = 0
    return TensorDict({'actor': frames.flatten(1)}, batch_size=[8])


def model(obs):
    return HistoryGaussianStudentMLP(
        obs, {'actor': ['actor']}, 'actor', 6, hidden_dims=(16,), obs_normalization=True,
    )


def test_upstream_history_is_chronological_and_latest_frame_routes():
    buffer = CircularBuffer(max_len=5, batch_size=8, device='cpu')
    frames = observations()['actor'].reshape(8, 5, 56)
    for frame in frames.unbind(1):
        buffer.append(frame)
    assert torch.equal(buffer.buffer, frames)
    obs = TensorDict({'actor': buffer.buffer.flatten(1)}, batch_size=[8])
    actor = model(obs)
    actor.update_normalization(obs)
    router = cast(HistoryTaskRoutedNetwork, actor.mlp)
    packed = router(actor.get_latent(obs))
    assert torch.equal(packed[:, 1, 0], frames[:, -1, 39].round())
    for task, width in enumerate((6, 3, 6, 3)):
        head = cast(nn.Sequential, router.heads[task])
        mlp = cast(nn.Sequential, head[1])
        assert cast(nn.Linear, mlp[0]).in_features == 305
        assert cast(nn.Linear, mlp[-1]).out_features == width


def test_encoding_preserves_per_frame_category_and_recovery_residual():
    raw = observations()['actor']
    encoded = HistoryTaskEncoding(5)(raw).reshape(8, 5, 61)
    frames = raw.reshape(8, 5, 56)
    assert torch.equal(encoded[:, :, 39:43], torch.nn.functional.one_hot(frames[:, :, 39].round().long(), 4).float())
    assert torch.equal(encoded[:, :, 43], frames[:, :, 39] - frames[:, :, 39].round())
    assert torch.equal(encoded[:, :, 50], torch.zeros(8, 5))
    assert torch.equal(encoded[:, :, 51], torch.ones(8, 5))


def test_normalization_separates_task_frames_and_preserves_codes():
    normalizer = HistoryTaskNormalization(5)
    frames = torch.zeros(2, 5, 56)
    frames[0, :, 39] = torch.tensor([0., 1., 0., 1., 0.])
    frames[1, :, 39] = 2
    frames[:, :, 0] = frames[:, :, 39] * 100 + 7
    frames[:, :, 38] = -1
    frames[:, :, 46] = .4
    raw = frames.flatten(1)
    normalizer.update(raw)
    moments = normalizer.frame_normalizer
    assert torch.equal(moments.count, torch.tensor([3, 2, 5, 0]))
    assert torch.equal(moments._mean[:3, 0], torch.tensor([7., 107., 207.]))
    assert torch.equal(moments._mean[3], torch.zeros(56))
    normalized = normalizer(raw).reshape(2, 5, 56)
    assert torch.equal(normalized[:, :, 0], torch.zeros(2, 5))
    assert torch.equal(normalized[:, :, 38:40], frames[:, :, 38:40])
    assert torch.equal(normalized[:, :, 46], frames[:, :, 46])
    normalizer.eval()
    normalizer.update(raw)
    assert torch.equal(moments.count, torch.tensor([3, 2, 5, 0]))


def test_mirrored_samples_finite_backward_and_jit_roundtrip():
    obs = observations()
    actor = model(obs)
    actor.update_normalization(obs)
    deterministic = actor(obs)
    sampled = actor(obs, stochastic_output=True)
    for task in (1, 3):
        assert torch.equal(sampled[task::4], expand_mirror(sampled[task::4, :3]))
        assert torch.equal(deterministic[task::4], expand_mirror(deterministic[task::4, :3]))
    loss = -actor.get_output_log_prob(sampled.detach()).mean() - .01 * actor.output_entropy.mean()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in actor.parameters() if parameter.grad is not None)
    actor.clear_inactive_distribution_gradients({1, 3})
    distribution = cast(GaussianTaskDistribution, actor.distribution)
    for task in (0, 2):
        assert all(parameter.grad is None for parameter in distribution.gaussians[task].parameters())
    actor.eval()
    expected = actor(obs)
    scripted = torch.jit.script(actor.as_jit())
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    assert torch.equal(torch.jit.load(buffer)(obs['actor']), expected)


@pytest.mark.parametrize('width,normalize', [(279, True), (280, False)])
def test_invalid_history_configuration_fails(width, normalize):
    with pytest.raises(ValueError):
        HistoryGaussianStudentMLP(
            {'actor': torch.zeros(4, width)}, {'actor': ['actor']}, 'actor', 6,
            hidden_dims=(8,), obs_normalization=normalize,
        )
