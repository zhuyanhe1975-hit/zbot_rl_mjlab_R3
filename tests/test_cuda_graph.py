import pytest
import torch

from zbot_rl_mjlab.cuda_graph import FixedTensorGraph


def test_reject_cpu():
    with pytest.raises(ValueError, match='CUDA'):
        FixedTensorGraph(lambda x: x + 1, torch.zeros(2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_replay_updates_inputs_parameters_and_owns_output():
    model = torch.nn.Linear(4, 3).cuda().eval()
    value = torch.randn(8, 4, device='cuda')
    graph = FixedTensorGraph(model, value)
    previous = graph(value)
    saved = previous.clone()
    next_value = value + 2
    with torch.no_grad():
        model.weight.add_(1)
    actual = graph(next_value)
    torch.testing.assert_close(actual, model(next_value))
    torch.testing.assert_close(previous, saved)
    assert not actual.requires_grad
    assert not model.training
    with pytest.raises(ValueError, match='shape'):
        graph(value[:2])
    with pytest.raises(ValueError, match='dtype'):
        graph(value.double())
