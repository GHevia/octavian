"""Objective declarations for missions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Objective:
    """Base class for user-facing mission objectives."""

    kind: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("Objective weight must be non-negative.")


@dataclass(frozen=True, slots=True)
class MinimizeDeltaV(Objective):
    """Minimize the sum of impulsive maneuver magnitudes."""

    kind: str = "delta_v"


@dataclass(frozen=True, slots=True)
class MinimizeTime(Objective):
    """Minimize total time of flight."""

    kind: str = "time"


def minimize_total_delta_v(weight: float = 1.0) -> MinimizeDeltaV:
    """Create a total delta-v objective.

    Args:
        weight: Scalar weight applied by the solver backend.

    Returns:
        A delta-v objective declaration.
    """
    return MinimizeDeltaV(weight=float(weight))


def minimize_total_time(weight: float = 1.0) -> MinimizeTime:
    """Create a total time-of-flight objective.

    Args:
        weight: Scalar weight applied by the solver backend.

    Returns:
        A time objective declaration.
    """
    return MinimizeTime(weight=float(weight))
