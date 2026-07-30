"""Phase definitions for mission composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .control import ThrustControl
from .events import BoundaryEvent
from .links import Link
from .links import continuous as continuous_link
from .links import impulsive as impulsive_link
from .models import Dynamics
from .spacecraft import Spacecraft
from .specs import BoundaryState


@dataclass(slots=True)
class Phase:
    """One dynamics segment in a composable mission.

    ``initial_state`` and ``final_state`` are seed anchors unless matching
    constraints are declared; this distinction permits free-time and partial
    terminal targets. Relative-element phases may therefore keep convenient
    RIC seed anchors while applying native ROE constraints.

    Args:
        name: Human-readable phase identifier.
        mode: Coast, relative-coast, or powered phase semantics.
        spacecraft: Vehicle configuration or registry name.
        dynamics: Translational dynamics configuration.
        initial_state: Optional Cartesian state used to seed the phase.
        final_state: Optional Cartesian state used to seed the phase.
        constraints: Boundary, path, geometry, and element constraints.
        tof_bounds_s: Lower and upper phase-end time or duration.
        initial_guess: Optional specialized guess configuration.
        thrust_control: Optional finite-thrust direction or kinematic-attitude
            representation. Linked phases inherit the previous declaration.
    """

    name: str = "phase"
    mode: str = "coast"
    spacecraft: Spacecraft | str | None = None
    dynamics: Dynamics | None = None

    initial_state: BoundaryState | None = None
    final_state: BoundaryState | None = None

    epoch: str | None = None

    constraints: list[Any] = field(default_factory=list)
    events: list[BoundaryEvent] = field(default_factory=list)
    variables: list[Any] = field(default_factory=list)

    previous: Phase | None = None
    link: Link | None = None

    tof_bounds_s: tuple[float, float] | None = None
    tof_is_relative: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    initial_guess: Any | None = None
    thrust_control: ThrustControl | None = None

    def inherit_defaults(self) -> None:
        """Inherit spacecraft, dynamics, and default link from the previous phase."""
        if self.previous is None:
            return

        if self.spacecraft is None:
            self.spacecraft = self.previous.spacecraft
        if self.dynamics is None:
            self.dynamics = self.previous.dynamics
        if self.thrust_control is None:
            self.thrust_control = self.previous.thrust_control

        if self.link is None:
            normalized_mode = (self.mode or "").lower()
            self.link = (
                impulsive_link()
                if normalized_mode in ("rendezvous", "transfer")
                else continuous_link()
            )

    def has_impulse(self, where: str) -> bool:
        """Return whether the phase declares an impulse at a boundary.

        Args:
            where: Boundary name such as ``"Front"`` or ``"Back"``.

        Returns:
            ``True`` if the phase has an impulse event at that boundary.
        """
        normalized_where = (where or "").strip().lower()
        boundary = (
            "Front"
            if normalized_where in ("front", "start", "initial", "t0")
            else "Back"
            if normalized_where in ("back", "end", "final", "tf")
            else None
        )
        if boundary is None:
            raise ValueError("where must be 'front' or 'back'.")
        return any(
            getattr(event, "kind", "") == "impulse" and getattr(event, "where", "") == boundary
            for event in self.events
        )

    def validate(self) -> None:
        """Validate that the phase has the required data to solve.

        Raises:
            ValueError: If the phase is missing dynamics, spacecraft, or valid
                time-of-flight bounds.
        """
        self.inherit_defaults()
        if self.dynamics is None:
            raise ValueError(f"Phase {self.name!r} is missing dynamics.")
        if self.spacecraft is None:
            raise ValueError(f"Phase {self.name!r} is missing spacecraft.")
        if self.tof_bounds_s is not None:
            t_min_s, t_max_s = map(float, self.tof_bounds_s)
            if not (t_max_s > t_min_s >= 0.0):
                raise ValueError(f"Phase {self.name!r} has invalid tof_bounds_s.")


def state(r_m: Sequence[float], v_mps: Sequence[float]) -> BoundaryState:
    """Create a boundary state from position and velocity vectors.

    Args:
        r_m: Position vector in meters.
        v_mps: Velocity vector in meters per second.

    Returns:
        A boundary-state object with reshaped ``(3,)`` vectors.
    """
    return BoundaryState(
        np.asarray(r_m, dtype=float).reshape(3),
        np.asarray(v_mps, dtype=float).reshape(3),
    )
