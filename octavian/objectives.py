from __future__ import annotations

"""Objective definitions.

Objectives are explicit, user-facing objects. They are collected on `Mission`
and mapped to backend solver objective terms.

v0.x focuses on impulsive rendezvous, so the built-in objectives are:
  - Minimize total Δv (default)
  - Minimize total time (optional)

Future versions can add control effort, propellant, pointing, etc.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Objective:
    kind: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("Objective weight must be non-negative")


@dataclass(frozen=True, slots=True)
class MinimizeDeltaV(Objective):
    """Minimize the sum of Δv magnitudes for impulsive maneuvers."""
    kind: str = "delta_v"


@dataclass(frozen=True, slots=True)
class MinimizeTime(Objective):
    """Minimize total time-of-flight (weighted)."""
    kind: str = "time"


def minimize_total_delta_v(weight: float = 1.0) -> MinimizeDeltaV:
    return MinimizeDeltaV(weight=float(weight))


def minimize_total_time(weight: float = 1.0) -> MinimizeTime:
    return MinimizeTime(weight=float(weight))
