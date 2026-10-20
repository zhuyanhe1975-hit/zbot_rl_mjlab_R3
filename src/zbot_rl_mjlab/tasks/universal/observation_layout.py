"""Shared offsets for compact teacher observations and archived checkpoints."""

import torch

PRIVILEGED_OMNI_OBS_DIM = 56
LEGACY_PRIVILEGED_OMNI_OBS_DIM = 59


def _dimension(observations: torch.Tensor) -> int:
    if observations.ndim < 1 or observations.shape[-1] not in (
        PRIVILEGED_OMNI_OBS_DIM, LEGACY_PRIVILEGED_OMNI_OBS_DIM,
    ):
        raise ValueError('teacher observations must have trailing dimension 56 or 59')
    return observations.shape[-1]


def teacher_task_ids(observations: torch.Tensor) -> torch.Tensor:
    """Decode latched task IDs without copying CUDA tensors to the host."""
    if _dimension(observations) == PRIVILEGED_OMNI_OBS_DIM:
        return observations[..., 39].round().long()
    return observations[..., 39:43].argmax(dim=-1)


def teacher_motion_fields(observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return effective command, actual site-frame velocity, pending request."""
    if _dimension(observations) == PRIVILEGED_OMNI_OBS_DIM:
        return observations[..., 40:43], observations[..., 50:53], observations[..., 53:56]
    return observations[..., 43:46], observations[..., 53:56], observations[..., 56:59]
