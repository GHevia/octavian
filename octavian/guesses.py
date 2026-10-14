"""Typed initial-guess declarations for mission phases."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class TrajectoryGuess:
    """Provide an explicit dense state/time history to seed a phase.

    Rows contain at least ``[x, y, z, vx, vy, vz, time]``. Extra columns are
    retained for phase formulations that carry additional states or controls.
    Times must be strictly increasing.
    """

    rows: NDArray[np.float64]

    def __init__(self, rows: ArrayLike) -> None:
        trajectory = np.asarray(rows, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 7:
            raise ValueError(
                "TrajectoryGuess rows must contain at least two "
                "[x, y, z, vx, vy, vz, time] samples."
            )
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("TrajectoryGuess rows must be finite.")
        if not np.all(np.diff(trajectory[:, 6]) > 0.0):
            raise ValueError("TrajectoryGuess times must be strictly increasing.")
        object.__setattr__(self, "rows", trajectory.copy())


@dataclass(frozen=True, slots=True)
class LowThrustSpiralGuess:
    """Configure a near-circular tangential spiral initial guess.

    Args:
        throttle: Constant throttle magnitude used only to integrate the seed.
        direction: ``"auto"`` chooses prograde for raising and retrograde for
            lowering. The direction may also be specified explicitly.
        steps_per_orbit: Minimum RK4 integration steps per initial orbit.
        time_scale: Multiplier on the circular-spiral burn-time estimate before
            it is clamped to the phase time bounds.
    """

    throttle: float = 0.8
    direction: str = "auto"
    steps_per_orbit: int = 120
    time_scale: float = 1.0

    def __post_init__(self) -> None:
        throttle = float(self.throttle)
        direction = str(self.direction).strip().lower()
        steps_per_orbit = int(self.steps_per_orbit)
        time_scale = float(self.time_scale)
        if not isfinite(throttle) or not (0.0 < throttle <= 1.0):
            raise ValueError("LowThrustSpiralGuess.throttle must be in (0, 1].")
        if direction not in ("auto", "prograde", "retrograde"):
            raise ValueError(
                "LowThrustSpiralGuess.direction must be 'auto', 'prograde', or 'retrograde'."
            )
        if steps_per_orbit < 20:
            raise ValueError("LowThrustSpiralGuess.steps_per_orbit must be at least 20.")
        if not isfinite(time_scale) or time_scale <= 0.0:
            raise ValueError("LowThrustSpiralGuess.time_scale must be positive.")
        object.__setattr__(self, "throttle", throttle)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "steps_per_orbit", steps_per_orbit)
        object.__setattr__(self, "time_scale", time_scale)


def low_thrust_spiral(
    *,
    throttle: float = 0.8,
    direction: str = "auto",
    steps_per_orbit: int = 120,
    time_scale: float = 1.0,
) -> LowThrustSpiralGuess:
    """Create a tangential low-thrust spiral seed declaration."""
    return LowThrustSpiralGuess(
        throttle=float(throttle),
        direction=str(direction),
        steps_per_orbit=int(steps_per_orbit),
        time_scale=float(time_scale),
    )


def trajectory(rows: ArrayLike) -> TrajectoryGuess:
    """Create an explicit dense trajectory seed declaration."""
    return TrajectoryGuess(rows)


@dataclass(frozen=True, slots=True)
class LowThrustQLawGuess:
    """Equinoctial proximity-quotient steering for a two-body orbit seed.

    Targets the orbit of ``Phase.final_state`` (including its plane), leaving
    orbital phase free. This numerical Q-law variant uses sampled maximum
    Gauss element rates, without periapsis penalties or efficiency coasting.
    It supports elliptic orbits away from the retrograde equinoctial singularity.
    Tolerances terminate seed steering; they do not impose solver constraints.
    """

    throttle: float = 0.8
    steps_per_orbit: int = 120
    orbit_samples: int = 32
    semi_major_axis_rtol: float = 0.001
    eccentricity_tol: float = 0.005
    plane_tol_deg: float = 0.1

    def __post_init__(self) -> None:
        if not isfinite(self.throttle) or not 0.0 < self.throttle <= 1.0:
            raise ValueError("LowThrustQLawGuess.throttle must be in (0, 1].")
        for name, minimum in (("steps_per_orbit", 20), ("orbit_samples", 16)):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"LowThrustQLawGuess.{name} must be an integer >= {minimum}.")
        for name in ("semi_major_axis_rtol", "eccentricity_tol", "plane_tol_deg"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"LowThrustQLawGuess.{name} must be finite and positive.")


def low_thrust_qlaw(
    *,
    throttle: float = 0.8,
    steps_per_orbit: int = 120,
    orbit_samples: int = 32,
    semi_major_axis_rtol: float = 0.001,
    eccentricity_tol: float = 0.005,
    plane_tol_deg: float = 0.1,
) -> LowThrustQLawGuess:
    """Seed simultaneous orbit-size, eccentricity-vector, and plane changes.

    Use on a single inertial ``low_thrust`` phase with vector thrust control.
    Specify the desired orbit using ``final_state`` and independently declare
    terminal constraints; this guess does not constrain the optimized orbit.
    """
    return LowThrustQLawGuess(
        throttle=throttle,
        steps_per_orbit=steps_per_orbit,
        orbit_samples=orbit_samples,
        semi_major_axis_rtol=semi_major_axis_rtol,
        eccentricity_tol=eccentricity_tol,
        plane_tol_deg=plane_tol_deg,
    )
