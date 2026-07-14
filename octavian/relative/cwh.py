"""Clohessy-Wiltshire dynamics for motion near a circular chief orbit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..bodies import EARTH, CelestialBody
from ..coordinates import CoordinateFrame, SolverScaling, lvlh

StateVector = NDArray[np.float64]
StateMatrix = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ClohessyWiltshire:
    """Linear time-invariant relative dynamics about a circular chief.

    States use a chief-centered LVLH/RTN frame with radial x, along-track y,
    and orbit-normal z. ``reference_length_m`` controls numerical scaling; it
    is not a validity limit for the linear model.
    """

    mean_motion_radps: float
    chief_orbit_radius_m: float | None = None
    chief_name: str = "chief"
    reference_length_m: float = 1_000.0

    def __post_init__(self) -> None:
        if float(self.mean_motion_radps) <= 0.0:
            raise ValueError("ClohessyWiltshire.mean_motion_radps must be positive")
        if self.chief_orbit_radius_m is not None and float(self.chief_orbit_radius_m) <= 0.0:
            raise ValueError("ClohessyWiltshire.chief_orbit_radius_m must be positive")
        if float(self.reference_length_m) <= 0.0:
            raise ValueError("ClohessyWiltshire.reference_length_m must be positive")
        if not str(self.chief_name).strip():
            raise ValueError("ClohessyWiltshire.chief_name must not be empty")
        object.__setattr__(self, "mean_motion_radps", float(self.mean_motion_radps))
        if self.chief_orbit_radius_m is not None:
            object.__setattr__(self, "chief_orbit_radius_m", float(self.chief_orbit_radius_m))
        object.__setattr__(self, "reference_length_m", float(self.reference_length_m))

    @classmethod
    def from_circular_orbit(
        cls,
        radius_m: float,
        *,
        body: CelestialBody = EARTH,
        chief_name: str = "chief",
        reference_length_m: float = 1_000.0,
    ) -> ClohessyWiltshire:
        """Construct CWH parameters from a circular chief-orbit radius."""
        radius = float(radius_m)
        if radius <= 0.0:
            raise ValueError("radius_m must be positive")
        mean_motion = float(np.sqrt(float(body.mu_m3ps2) / radius**3))
        return cls(
            mean_motion_radps=mean_motion,
            chief_orbit_radius_m=radius,
            chief_name=chief_name,
            reference_length_m=reference_length_m,
        )

    @property
    def frame(self) -> CoordinateFrame:
        """Return the LVLH frame in which this model's states are expressed."""
        return lvlh(self.chief_name)

    @property
    def scaling(self) -> SolverScaling:
        """Return physically consistent characteristic units for this model."""
        return SolverScaling(
            length_m=self.reference_length_m,
            velocity_mps=self.mean_motion_radps * self.reference_length_m,
            time_s=1.0 / self.mean_motion_radps,
        )


def cwh_derivative(state: ArrayLike, mean_motion_radps: float) -> StateVector:
    """Evaluate the six-state unforced CWH differential equation."""
    x, y, z, xdot, ydot, zdot = np.asarray(state, dtype=float).reshape(6)
    del y  # Position y is absent from the linearized acceleration expression.
    n = float(mean_motion_radps)
    if n <= 0.0:
        raise ValueError("mean_motion_radps must be positive")
    return np.asarray(
        [
            xdot,
            ydot,
            zdot,
            3.0 * n**2 * x + 2.0 * n * ydot,
            -2.0 * n * xdot,
            -(n**2) * z,
        ],
        dtype=float,
    )


def cwh_state_transition(dt_s: float, mean_motion_radps: float) -> StateMatrix:
    """Return the analytic 6x6 CWH state-transition matrix."""
    n = float(mean_motion_radps)
    if n <= 0.0:
        raise ValueError("mean_motion_radps must be positive")
    t = float(dt_s)
    nt = n * t
    c = float(np.cos(nt))
    s = float(np.sin(nt))
    return np.asarray(
        [
            [4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * (1.0 - c) / n, 0.0],
            [6.0 * (s - nt), 1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, c, 0.0, 0.0, s / n],
            [3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0],
            [-6.0 * n * (1.0 - c), 0.0, 0.0, -2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, -n * s, 0.0, 0.0, c],
        ],
        dtype=float,
    )


def propagate_cwh(state: ArrayLike, dt_s: float, mean_motion_radps: float) -> StateVector:
    """Propagate one relative state with the analytic CWH solution."""
    return cwh_state_transition(dt_s, mean_motion_radps) @ np.asarray(
        state, dtype=float
    ).reshape(6)


def cwh_rendezvous_velocity(
    initial_position_m: ArrayLike,
    final_position_m: ArrayLike,
    tof_s: float,
    mean_motion_radps: float,
) -> StateVector:
    """Return the initial velocity that connects two CWH positions in time."""
    phi = cwh_state_transition(tof_s, mean_motion_radps)
    phi_rr = phi[0:3, 0:3]
    phi_rv = phi[0:3, 3:6]
    rhs = np.asarray(final_position_m, dtype=float).reshape(3) - phi_rr @ np.asarray(
        initial_position_m, dtype=float
    ).reshape(3)
    try:
        return np.linalg.solve(phi_rv, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The requested CWH transfer time is singular for position targeting; "
            "choose a different time of flight."
        ) from exc
