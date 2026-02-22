"""Phase linking semantics.

A `Link` defines which quantities are constrained to be continuous across a phase
boundary when `Phase(previous=...)` is used.

In v0.x, links are primarily used to express *impulsive* rendezvous structure:
  - position and time are continuous
  - velocity may be discontinuous (a Δv at the boundary)

This module is intentionally small; it is an API/semantics layer that Octavian's
current impulsive solvers can map onto.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Link:
    """Link configuration between two phases."""

    kind: str = "continuous"  # "continuous" | "impulsive"
    dv_max_mps: float | None = None  # optional soft cap (see solvers)
    name: str = "link"

    def is_impulsive(self) -> bool:
        return self.kind.lower() == "impulsive"

    def is_continuous(self) -> bool:
        return self.kind.lower() == "continuous"


def continuous(*, name: str = "continuous") -> Link:
    """Enforce continuity of (R, V, t) across the phase boundary."""
    return Link(kind="continuous", name=name)


def impulsive(*, dv_max_mps: float | None = None, name: str = "impulsive") -> Link:
    """Enforce continuity of (R, t), allow a velocity jump Δv at the boundary.

    Notes
    -----
    In v0.x solvers, Δv is represented implicitly by allowing front/back boundary
    velocities to differ and optionally adding a Δv objective term. If dv_max_mps
    is provided, current solvers treat it as a *soft* cap (penalty) unless/until
    hard norm bounds are added.
    """
    return Link(kind="impulsive", dv_max_mps=dv_max_mps, name=name)
