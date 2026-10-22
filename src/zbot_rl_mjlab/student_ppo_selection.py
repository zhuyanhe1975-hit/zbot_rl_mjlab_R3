"""Validation-only checkpoint selection and baseline regression protection."""

import math
from collections.abc import Mapping
from typing import Any

SCORE_METRICS = ('biped_moving_success', 'snake_moving_success',
                 'descent_handoff_fraction', 'ascent_handoff_fraction')
# Match the evaluator's signed motion cases without importing its simulator.
SIGNED_METRICS = tuple(
    f'{mode}_{motion}_success'
    for mode in ('biped', 'snake')
    for motion in ('forward', 'backward', 'left', 'right', 'yaw_left',
                   'yaw_right', 'mixed_positive', 'mixed_negative')
)
REQUIRED_METRICS = (*SIGNED_METRICS, *SCORE_METRICS,
                    'biped_stop_success', 'snake_stop_success',
                    'ascent_success', 'descent_success')


class StudentPPOCheckpointSelector:
    """Rank eligible policies by worst skill, then mean skill success.

    Tolerance is an absolute success-fraction drop from the initial baseline;
    it is not relative to the last validation or a moving best checkpoint.
    Invalid observations count toward the same consecutive failure patience.
    """

    def __init__(self, baseline: Mapping[str, float], *, tolerance: float = .25,
                 patience: int = 3):
        if isinstance(tolerance, bool) or not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
            raise ValueError('validation tolerance must be finite and in [0, 1]')
        if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
            raise ValueError('validation patience must be a positive integer')
        self.tolerance, self.patience = tolerance, patience
        self.baseline = self._metrics(baseline)
        self.best_metrics = self.baseline.copy()
        self.best_score = self._score(self.best_metrics)
        self.consecutive_failures = 0

    @staticmethod
    def _metrics(values: Mapping[str, float]) -> dict[str, float]:
        result = {}
        for key in REQUIRED_METRICS:
            try:
                value = float(values[key])
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f'missing or invalid validation metric: {key}') from error
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'validation metric must be finite and in [0, 1]: {key}')
            result[key] = value
        return result

    @staticmethod
    def _score(values: Mapping[str, float]) -> tuple[float, float]:
        skills = [values[key] for key in SCORE_METRICS]
        return min(skills), sum(skills) / len(skills)

    def _degraded(self, values: Mapping[str, float]) -> list[str]:
        return [key for key in REQUIRED_METRICS
                if values[key] < self.baseline[key] - self.tolerance]

    def observe(self, values: Mapping[str, float]) -> dict[str, Any]:
        """Observe one validation; only eligible strict improvements become best."""
        score = None
        degraded: list[str] = []
        improved = False
        try:
            metrics = self._metrics(values)
        except ValueError as error:
            eligible, reason = False, str(error)
        else:
            score = self._score(metrics)
            degraded = self._degraded(metrics)
            eligible = not degraded
            reason = 'eligible' if eligible else 'baseline regression: ' + ', '.join(degraded)
            improved = eligible and score > self.best_score
            if improved:
                self.best_metrics, self.best_score = metrics, score
        self.consecutive_failures = 0 if eligible else self.consecutive_failures + 1
        return {'eligible': eligible, 'improved': improved,
                'should_stop': self.consecutive_failures >= self.patience,
                'score': list(score) if score is not None else None,
                'reason': reason, 'degraded_metrics': degraded,
                'consecutive_failures': self.consecutive_failures}

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-serializable state; callers own checkpoint/file I/O."""
        return {'baseline': self.baseline.copy(), 'best_metrics': self.best_metrics.copy(),
                'tolerance': self.tolerance, 'patience': self.patience,
                'consecutive_failures': self.consecutive_failures}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> 'StudentPPOCheckpointSelector':
        selector = cls(state['baseline'], tolerance=state['tolerance'], patience=state['patience'])
        best = selector._metrics(state['best_metrics'])
        score = selector._score(best)
        failures = state['consecutive_failures']
        if selector._degraded(best) or score < selector.best_score:
            raise ValueError('restored best checkpoint violates baseline selection constraints')
        if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
            raise ValueError('restored consecutive failures must be a nonnegative integer')
        selector.best_metrics, selector.best_score = best, score
        selector.consecutive_failures = failures
        return selector
