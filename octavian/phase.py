"""Phase definition.

`Phase` is the unit of composition for a Mission.

In v0.x, Octavian ships only a small set of solvers (e.g., impulsive rendezvous).
Phase objects still matter because they:
  - keep user scripts readable (config-like)
  - provide a home for dynamics/scaling/constraints
  - enable auto-linking via `previous=...`

This module also hosts the small `state(...)` convenience helper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .events import BoundaryEvent
from .links import Link
from .links import continuous as continuous_link
from .links import impulsive as impulsive_link
from .models import Dynamics
from .spacecraft import Spacecraft
from .specs import BoundaryState


@dataclass(slots=True)
class Phase:
    name: str = "phase"
    mode: str = "coast"  # "coast" | "burn" | "rendezvous" (semantics layer)
    spacecraft: Spacecraft | str | None = None
    dynamics: Dynamics | None = None

    initial_state: BoundaryState | None = None
    final_state: BoundaryState | None = None

    epoch: str | None = None

    # User-facing declarations
    constraints: list[Any] = field(default_factory=list)
    events: list[BoundaryEvent] = field(default_factory=list)

    # Decision variables / structure (composable missions)
    variables: list[Any] = field(default_factory=list)

    # Linking to a previous phase
    previous: Phase | None = None
    link: Link | None = None

    # Time-of-flight bounds; interpreted as absolute Back-time bounds by default.
    # Set tof_is_relative=True to treat bounds as per-phase durations.
    tof_bounds_s: tuple[float, float] | None = None
    tof_is_relative: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def inherit_defaults(self) -> None:
        if self.previous is None:
            return
        if self.spacecraft is None:
            self.spacecraft = self.previous.spacecraft
        if self.dynamics is None:
            self.dynamics = self.previous.dynamics

        # Default link choice: keep scripts simple.
        if self.link is None:
            if (self.mode or "").lower() in ("rendezvous", "transfer"):
                self.link = impulsive_link()
            else:
                self.link = continuous_link()

    def has_impulse(self, where: str) -> bool:
        w = (where or "").strip().lower()
        loc = (
            "Front"
            if w in ("front", "start", "initial", "t0")
            else "Back" if w in ("back", "end", "final", "tf") else None
        )
        if loc is None:
            raise ValueError("where must be 'front' or 'back'")
        return any(
            getattr(ev, "kind", "") == "impulse" and getattr(ev, "where", "") == loc
            for ev in self.events
        )

    def validate(self) -> None:
        self.inherit_defaults()
        if self.dynamics is None:
            raise ValueError(f"Phase {self.name!r} is missing dynamics")
        if self.spacecraft is None:
            raise ValueError(f"Phase {self.name!r} is missing spacecraft")
        if self.tof_bounds_s is not None:
            a, b = map(float, self.tof_bounds_s)
            if not (b > a >= 0.0):
                raise ValueError(f"Phase {self.name!r} has invalid tof_bounds_s")


def state(r_m: Sequence[float], v_mps: Sequence[float]) -> BoundaryState:
    """Small helper to build a `BoundaryState`."""
    return BoundaryState(np.asarray(r_m, float).reshape(3), np.asarray(v_mps, float).reshape(3))
