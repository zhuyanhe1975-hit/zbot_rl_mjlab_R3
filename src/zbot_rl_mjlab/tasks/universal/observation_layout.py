"""Shared offsets for compact teacher observations and archived checkpoints."""

import torch

PRIVILEGED_OMNI_OBS_DIM = 56
LEGACY_PRIVILEGED_OMNI_OBS_DIM = 59
# Current site snake removes compact heading at 46: task/command stay at 39
# and 40:43, actual velocity becomes 49:52, pending request 52:55,
# normalized tracking error 55:58, and site-local gravity 58:61.
SITE_SNAKE_OBS_DIM = 61
# Archived site schema has heading sin/cos 46:48, actual velocity 51:54,
# request 54:57, tracking error 57:60, and gravity 60:63.
LEGACY_SITE_SNAKE_OBS_DIM = 63


def _dimension(observations: torch.Tensor) -> int:
    if observations.ndim < 1 or observations.shape[-1] not in (
        PRIVILEGED_OMNI_OBS_DIM, LEGACY_PRIVILEGED_OMNI_OBS_DIM, SITE_SNAKE_OBS_DIM, LEGACY_SITE_SNAKE_OBS_DIM,
    ):
        raise ValueError('teacher observations must have trailing dimension 56, 59, 61 or 63')
    return observations.shape[-1]


def teacher_task_ids(observations: torch.Tensor) -> torch.Tensor:
    """Decode latched task IDs without copying CUDA tensors to the host."""
    if _dimension(observations) != LEGACY_PRIVILEGED_OMNI_OBS_DIM:
        return observations[..., 39].round().long()
    return observations[..., 39:43].argmax(dim=-1)


def teacher_motion_fields(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return effective command, actual tracking velocity, pending request."""
    dimension = _dimension(observations)
    if dimension == SITE_SNAKE_OBS_DIM:
        return observations[..., 40:43], observations[..., 49:52], observations[..., 52:55]
    if dimension == LEGACY_SITE_SNAKE_OBS_DIM:
        return observations[..., 40:43], observations[..., 51:54], observations[..., 54:57]
    if dimension == PRIVILEGED_OMNI_OBS_DIM:
        return observations[..., 40:43], observations[..., 50:53], observations[..., 53:56]
    return observations[..., 43:46], observations[..., 53:56], observations[..., 56:59]
