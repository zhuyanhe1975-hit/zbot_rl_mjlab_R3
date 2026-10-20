"""Explicit requested tasks; physical observations only confirm completion."""

from enum import IntEnum

import torch


class Morphology(IntEnum):
    SNAKE = -1
    TRANSITION = 0
    BIPED = 1


class TaskState(IntEnum):
    BIPED_MOTION = 0
    DESCENDING = 1
    SNAKE_MOTION = 2
    ASCENDING = 3


class TaskStateMachine:
    """Batched latched tasks, changed only by reset, requests and settlement.

    ``source_morphology`` retains the last confirmed configuration during a
    transition. Reversing a pending request still requires settlement, even if
    the new goal is that source configuration. A fall cannot change the task.
    """

    def __init__(self, num_envs: int, device: torch.device | str):
        # These buffers may first be allocated by play under inference_mode,
        # then reused by keyboard callbacks outside that context.
        with torch.inference_mode(False):
            self.current_morphology = torch.full(
                (num_envs,), Morphology.SNAKE, dtype=torch.long, device=device,
            )
            self.source_morphology = self.current_morphology.clone()
            self.target_morphology = self.current_morphology.clone()
            self.task_state = torch.full(
                (num_envs,), TaskState.SNAKE_MOTION, dtype=torch.long, device=device,
            )

    @property
    def transition_mask(self) -> torch.Tensor:
        return self.current_morphology == Morphology.TRANSITION

    @staticmethod
    def _morphology(standing: torch.Tensor) -> torch.Tensor:
        return torch.where(standing, int(Morphology.BIPED), int(Morphology.SNAKE))

    @staticmethod
    def _task(standing: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
        return torch.where(
            transition,
            torch.where(standing, int(TaskState.ASCENDING), int(TaskState.DESCENDING)),
            torch.where(standing, int(TaskState.BIPED_MOTION), int(TaskState.SNAKE_MOTION)),
        )

    @torch.inference_mode()
    def reset(
        self,
        env_ids: torch.Tensor,
        initial_standing: torch.Tensor | None,
        target_standing: torch.Tensor,
        *,
        force_transition: bool = False,
    ) -> None:
        """Set known reset morphology and requested task for selected rows."""
        target = self._morphology(target_standing)
        initial = (
            torch.zeros_like(target) if initial_standing is None
            else self._morphology(initial_standing)
        )
        transition = torch.ones_like(target_standing) if force_transition else initial != target
        self.source_morphology[env_ids] = initial
        self.target_morphology[env_ids] = target
        self.current_morphology[env_ids] = torch.where(
            transition, int(Morphology.TRANSITION), initial,
        )
        self.task_state[env_ids] = self._task(target_standing, transition)

    @torch.inference_mode()
    def request(self, env_ids: torch.Tensor, target_standing: torch.Tensor) -> None:
        """Latch changed goals; repeated requests do not restart a task."""
        targets = self._morphology(target_standing)
        changed = self.target_morphology[env_ids] != targets
        selected = env_ids[changed]
        self.target_morphology[selected] = targets[changed]
        # Every changed request is a transition, including reversing one
        # already in progress. Only update_completion can enter motion.
        self.current_morphology[selected] = Morphology.TRANSITION
        self.task_state[selected] = self._task(
            target_standing[changed], torch.ones_like(target_standing[changed]),
        )

    @torch.inference_mode()
    def update_completion(self, settled: torch.Tensor) -> None:
        """Confirm pending goals after the caller's sustained readiness gate."""
        complete = self.transition_mask & settled
        self.current_morphology[complete] = self.target_morphology[complete]
        self.source_morphology[complete] = self.target_morphology[complete]
        standing = self.target_morphology[complete] == Morphology.BIPED
        self.task_state[complete] = self._task(standing, torch.zeros_like(standing))
