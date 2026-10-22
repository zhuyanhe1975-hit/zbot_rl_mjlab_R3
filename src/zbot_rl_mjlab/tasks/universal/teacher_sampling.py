"""Broad archived motion commands and stable training-role allocation."""

import math

import torch


def sample_motion_commands(
    standing: torch.Tensor,
    limits: tuple[float, float, float] = (.5, .2, .8),
    *, axis_min_fraction: float = .4, stop_probability: float = 1 / 8,
    x_only_probability: float | None = None,
) -> torch.Tensor:
    """By default sample stop (1/8), signed axes (3/8), and mixed motion (1/2).

    Biped mixed commands reproduce the successful omni-walking ranges. Snake
    ranges retain the universal teacher's planar command limits. Sampling does
    not shrink with tracking EMA: a stationary policy must see motion demand.
    Without an x override, stop probabilities retain the 3:4 single/mixed ratio.
    x_only_probability sets the absolute probability of signed pure-vx requests;
    remaining motion preserves the relative vy/wz/mixed coverage.
    """
    if standing.ndim != 1 or standing.dtype != torch.bool:
        raise ValueError('standing must be a one-dimensional bool tensor')
    if len(limits) != 3 or not all(math.isfinite(x) and x > 0 for x in limits):
        raise ValueError('limits must contain three finite positive values')
    if not math.isfinite(axis_min_fraction) or not 0 < axis_min_fraction <= 1:
        raise ValueError('axis_min_fraction must be in (0, 1]')
    if not math.isfinite(stop_probability) or not 0 <= stop_probability <= 1:
        raise ValueError('stop_probability must be finite and in [0, 1]')
    if x_only_probability is not None and (
        not math.isfinite(x_only_probability) or not 0 <= x_only_probability <= 1 - stop_probability
    ):
        raise ValueError('x_only_probability must be finite and in [0, 1-stop_probability]')
    n = standing.numel()
    device = standing.device
    biped_low = torch.tensor((-.3, -.2, -.5), device=device)
    biped_high = torch.tensor((.5, .2, .5), device=device)
    snake_high = torch.tensor((.25, .2, .8), device=device)
    low = torch.where(standing[:, None], biped_low, -snake_high)
    high = torch.where(standing[:, None], biped_high, snake_high)
    cap = torch.tensor(limits, device=device)
    low = torch.maximum(low, -cap)
    high = torch.minimum(high, cap)
    commands = low + torch.rand((n, 3), device=device) * (high - low)
    if x_only_probability is not None:
        category = torch.randint(4, 16, (n,), device=device)
        x_category = torch.randint(2, 4, (n,), device=device)
        draw = torch.rand(n, device=device)
        category = torch.where(draw < stop_probability + x_only_probability, x_category, category)
        category = torch.where(draw < stop_probability, 0, category)
    elif stop_probability == 1 / 8:
        category = torch.randint(16, (n,), device=device)
    else:
        category = torch.randint(2, 16, (n,), device=device)
        category = torch.where(torch.rand(n, device=device) < stop_probability, 0, category)
    axis = ((category - 2).clamp(0, 5) // 2)[:, None]
    positive = ((category - 2) % 2 == 0)[:, None]
    endpoint = torch.where(positive, high.gather(1, axis), low.gather(1, axis))
    magnitude = endpoint * (axis_min_fraction + (1 - axis_min_fraction) * torch.rand((n, 1), device=device))
    single = torch.zeros_like(commands).scatter(1, axis, magnitude)
    commands = torch.where((category < 8)[:, None], single, commands)
    return torch.where((category < 2)[:, None], 0., commands)


def training_roles(
    num_envs: int,
    ids: torch.Tensor,
    fractions: tuple[float, float, float, float] = (.4, .4, .1, .1),
) -> torch.Tensor:
    """Assign global IDs to biped, snake, ascent, and descent respectively.

    Rounded cumulative cutoffs preserve assignments across partial resets. Small
    scenes return -1, leaving role selection to the existing mixed sampler.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError('num_envs must be a positive integer')
    if len(fractions) != 4 or not all(math.isfinite(x) and x > 0 for x in fractions):
        raise ValueError('fractions must contain four finite positive values')
    if not math.isclose(sum(fractions), 1., abs_tol=1e-9):
        raise ValueError('fractions must sum to one')
    if ids.dtype != torch.long or ids.ndim != 1:
        raise ValueError('ids must be a one-dimensional long tensor')
    if bool(torch.any((ids < 0) | (ids >= num_envs))):
        raise ValueError('ids must be within the global environment count')
    if num_envs < 10:
        return torch.full_like(ids, -1)
    cutoffs = tuple(round(num_envs * sum(fractions[:i])) for i in range(1, 4))
    roles = torch.zeros_like(ids)
    for cutoff in cutoffs:
        roles += (ids >= cutoff).long()
    return roles
