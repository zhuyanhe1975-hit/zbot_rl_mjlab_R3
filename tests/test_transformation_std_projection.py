"""在 CPU 上使用真实 RSL 高斯分布和 torch 优化器。"""

import math

import pytest
import torch
from rsl_rl.modules import GaussianDistribution

from zbot_rl_mjlab.tasks.transformation.std_projection import ActionStdProjection


def parameter(distribution):
    return (
        distribution.log_std_param
        if distribution.std_type == "log"
        else distribution.std_param
    )


@pytest.mark.parametrize("std_type", ["scalar", "log"])
@pytest.mark.parametrize("boundary", ["below", "min", "max", "above"])
def test_projection_restores_entropy_and_log_prob_std_gradients(std_type, boundary):
    distribution = GaussianDistribution(6, std_type=std_type, std_range=(0.1, 1.0))
    value = parameter(distribution)
    bounds = distribution.log_std_range if std_type == "log" else distribution.std_range
    initial = {
        "below": bounds[0] - 0.1,
        "min": bounds[0],
        "max": bounds[1],
        "above": bounds[1] + 0.1,
    }[boundary]
    with torch.no_grad():
        value.fill_(initial)
    ActionStdProjection(distribution).project()
    assert (value > bounds[0]).all() and (value < bounds[1]).all()
    distribution.update(torch.zeros(3, 6))
    entropy_gradient = torch.autograd.grad(
        distribution.entropy.sum(), value, retain_graph=True
    )[0]
    log_prob_gradient = torch.autograd.grad(
        distribution.log_prob(torch.zeros(3, 6)).sum(), value
    )[0]
    assert torch.isfinite(entropy_gradient).all() and entropy_gradient.ne(0).all()
    assert torch.isfinite(log_prob_gradient).all() and log_prob_gradient.ne(0).all()


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_interior_unchanged_and_declared_ranges_preserved(std_type):
    distribution = GaussianDistribution(
        3, init_std=0.4, std_type=std_type, std_range=(0.1, 1.0)
    )
    before = parameter(distribution).detach().clone()
    declared = distribution.std_range.copy(), distribution.log_std_range.copy()
    projection = ActionStdProjection(distribution)
    projection.project()
    assert torch.equal(parameter(distribution), before)
    assert (distribution.std_range, distribution.log_std_range) == declared


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam])
@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_native_post_hook_projects_every_step_and_preserves_adam_state(
    optimizer_type, std_type
):
    distribution = GaussianDistribution(
        2, init_std=0.5, std_type=std_type, std_range=(0.1, 1.0)
    )
    value = parameter(distribution)
    bounds = distribution.log_std_range if std_type == "log" else distribution.std_range
    optimizer = optimizer_type(distribution.parameters(), lr=10.0)
    projection = ActionStdProjection(distribution)
    handle = projection.install(optimizer)
    for iteration in range(4):
        optimizer.zero_grad()
        loss = value.sum() * (1 if iteration < 2 else -1)
        loss.backward()
        optimizer.step()
        assert (value > bounds[0]).all() and (value < bounds[1]).all()
        assert optimizer.param_groups[0]["params"][0] is value
        if optimizer_type is torch.optim.Adam:
            assert optimizer.state[value]["step"] == iteration + 1
            assert optimizer.state[value]["exp_avg_sq"].gt(0).all()
    handle.remove()
    optimizer.zero_grad()
    value.sum().backward()
    optimizer.step()
    assert value.lt(bounds[0]).all()


def test_log_upper_zero_uses_normal_epsilon_instead_of_subnormal():
    distribution = GaussianDistribution(
        1, init_std=1.1, std_type="log", std_range=(0.1, 1.0)
    )
    ActionStdProjection(distribution).project()
    actual = distribution.log_std_param.detach()
    assert actual < 0
    assert actual.abs() >= torch.finfo(actual.dtype).tiny
    assert actual == -torch.finfo(actual.dtype).eps


@pytest.mark.parametrize("distribution", [None, object(), torch.nn.Identity()])
def test_unsupported_distribution_and_hook_are_noops(distribution):
    value = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([value], lr=0.1)
    projection = ActionStdProjection(distribution)
    projection.project()
    handle = projection.install(optimizer)
    value.backward()
    optimizer.step()
    assert value.item() == pytest.approx(0.9)
    handle.remove()


def test_no_representable_interior_fails_clearly():
    distribution = GaussianDistribution(1, std_range=(0.1, 1.0))
    distribution.std_range = [1.0, math.nextafter(1.0, math.inf)]
    with pytest.raises(ValueError, match="representable strict interior"):
        ActionStdProjection(distribution)


@pytest.mark.parametrize("bounds", [(1.0, 1.0), (2.0, 1.0), (0.1, math.inf)])
def test_invalid_declared_range_fails_clearly(bounds):
    distribution = GaussianDistribution(1)
    distribution.std_range = list(bounds)
    with pytest.raises(ValueError, match="finite and strictly ordered"):
        ActionStdProjection(distribution)


def test_log_already_interior_near_zero_remains_bitwise_unchanged():
    distribution = GaussianDistribution(1, std_type="log", std_range=(0.1, 1.0))
    with torch.no_grad():
        distribution.log_std_param.fill_(-1e-8)
    before = distribution.log_std_param.detach().clone()
    ActionStdProjection(distribution).project()
    assert torch.equal(distribution.log_std_param, before)


def test_adjacent_float32_bounds_have_no_strict_interior():
    distribution = GaussianDistribution(1)
    distribution.std_range = [1.0, 1.0 + torch.finfo(torch.float32).eps]
    with pytest.raises(ValueError, match="representable strict interior"):
        ActionStdProjection(distribution)


def test_single_representable_interior_value_is_supported():
    distribution = GaussianDistribution(1)
    step = torch.finfo(torch.float32).eps
    distribution.std_range = [1.0, 1.0 + 2 * step]
    ActionStdProjection(distribution).project()
    assert distribution.std_param.item() == 1.0 + step
