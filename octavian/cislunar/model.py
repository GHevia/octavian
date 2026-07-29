"""Physical system definition and equilibrium geometry for the CR3BP."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..bodies import EARTH, MOON, CelestialBody
from ..bodies import resolve as resolve_body
from ..coordinates import CoordinateFrame, SolverScaling
from ..coordinates.frames import synodic

Vector3 = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CR3BPSystem:
    """Circular restricted three-body problem physical configuration.

    The synodic frame is barycentric. Its +X axis points from ``primary`` to
    ``secondary``, +Z follows their orbital angular momentum, and it rotates at
    the pair's constant CR3BP mean motion.

    Args:
        primary: More massive body or catalog name.
        secondary: Less massive body or catalog name.
        separation_m: Constant circular separation between the bodies.
        name: Optional display name.
    """

    primary: CelestialBody | str
    secondary: CelestialBody | str
    separation_m: float
    name: str | None = None

    def __post_init__(self) -> None:
        primary = resolve_body(self.primary)
        secondary = resolve_body(self.secondary)
        separation = float(self.separation_m)
        if primary == secondary:
            raise ValueError("CR3BP primary and secondary must be different bodies")
        if not math.isfinite(separation) or separation <= 0.0:
            raise ValueError("CR3BP separation_m must be finite and positive")
        if secondary.mu_m3ps2 > primary.mu_m3ps2:
            raise ValueError("CR3BP primary must be at least as massive as secondary")
        object.__setattr__(self, "primary", primary)
        object.__setattr__(self, "secondary", secondary)
        object.__setattr__(self, "separation_m", separation)
        resolved_name = self.name or f"{primary.name}_{secondary.name}_cr3bp"
        normalized_name = str(resolved_name).strip()
        if not normalized_name:
            raise ValueError("CR3BP system name must not be empty")
        object.__setattr__(self, "name", normalized_name)

    @classmethod
    def earth_moon(
        cls,
        *,
        separation_m: float = 384_400_000.0,
    ) -> CR3BPSystem:
        """Return the conventional circular Earth–Moon CR3BP system."""
        return cls(EARTH, MOON, separation_m, name="earth_moon_cr3bp")

    @property
    def total_mu_m3ps2(self) -> float:
        """Return the sum of the primary and secondary gravitational parameters."""
        return self.primary.mu_m3ps2 + self.secondary.mu_m3ps2  # type: ignore[union-attr]

    @property
    def mass_parameter(self) -> float:
        """Return nondimensional secondary mass fraction ``mu``."""
        return self.secondary.mu_m3ps2 / self.total_mu_m3ps2  # type: ignore[union-attr]

    @property
    def mean_motion_radps(self) -> float:
        """Return constant synodic angular rate in radians per second."""
        return math.sqrt(self.total_mu_m3ps2 / self.separation_m**3)

    @property
    def period_s(self) -> float:
        """Return the circular primary-secondary period in seconds."""
        return 2.0 * math.pi / self.mean_motion_radps

    @property
    def time_scale_s(self) -> float:
        """Return the nondimensional CR3BP time unit in seconds."""
        return 1.0 / self.mean_motion_radps

    @property
    def velocity_scale_mps(self) -> float:
        """Return the nondimensional CR3BP velocity unit in m/s."""
        return self.separation_m * self.mean_motion_radps

    @property
    def frame(self) -> CoordinateFrame:
        """Return barycentric synodic frame metadata."""
        return synodic(self.primary.name, self.secondary.name)  # type: ignore[union-attr]

    @property
    def scaling(self) -> SolverScaling:
        """Return natural dimensional CR3BP solver units."""
        return SolverScaling(
            length_m=self.separation_m,
            velocity_mps=self.velocity_scale_mps,
            time_s=self.time_scale_s,
        )

    @property
    def primary_position_nondimensional(self) -> Vector3:
        """Return the primary's fixed nondimensional synodic position."""
        return np.asarray([-self.mass_parameter, 0.0, 0.0], dtype=float)

    @property
    def secondary_position_nondimensional(self) -> Vector3:
        """Return the secondary's fixed nondimensional synodic position."""
        return np.asarray([1.0 - self.mass_parameter, 0.0, 0.0], dtype=float)

    @property
    def primary_position_m(self) -> Vector3:
        """Return the primary's fixed dimensional synodic position."""
        return self.separation_m * self.primary_position_nondimensional

    @property
    def secondary_position_m(self) -> Vector3:
        """Return the secondary's fixed dimensional synodic position."""
        return self.separation_m * self.secondary_position_nondimensional

    def lagrange_points(
        self,
        *,
        dimensional: bool = True,
        tolerance: float = 1.0e-13,
    ) -> dict[str, Vector3]:
        """Return the five equilibrium points in the synodic frame.

        Args:
            dimensional: Return meters when true, nondimensional coordinates
                when false.
            tolerance: Collinear-root bisection tolerance in nondimensional
                distance.
        """
        if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
            raise ValueError("Lagrange-point tolerance must be finite and positive")
        mu = self.mass_parameter
        epsilon = max(1.0e-10, 10.0 * float(tolerance))
        x_l1 = _bisect_collinear(mu, -mu + epsilon, 1.0 - mu - epsilon, tolerance)
        x_l2 = _bisect_collinear(mu, 1.0 - mu + epsilon, 3.0, tolerance)
        x_l3 = _bisect_collinear(mu, -3.0, -mu - epsilon, tolerance)
        scale = self.separation_m if dimensional else 1.0
        return {
            "L1": scale * np.asarray([x_l1, 0.0, 0.0]),
            "L2": scale * np.asarray([x_l2, 0.0, 0.0]),
            "L3": scale * np.asarray([x_l3, 0.0, 0.0]),
            "L4": scale * np.asarray([0.5 - mu, math.sqrt(3.0) / 2.0, 0.0]),
            "L5": scale * np.asarray([0.5 - mu, -math.sqrt(3.0) / 2.0, 0.0]),
        }


def _collinear_equilibrium(x: float, mu: float) -> float:
    """Return the nondimensional collinear equilibrium equation."""
    primary_delta = x + mu
    secondary_delta = x - (1.0 - mu)
    return (
        x
        - (1.0 - mu) * primary_delta / abs(primary_delta) ** 3
        - mu * secondary_delta / abs(secondary_delta) ** 3
    )


def _bisect_collinear(
    mu: float,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    """Solve one bracketed collinear equilibrium root by bisection."""
    f_lower = _collinear_equilibrium(lower, mu)
    f_upper = _collinear_equilibrium(upper, mu)
    if f_lower * f_upper > 0.0:
        raise RuntimeError("Internal CR3BP Lagrange-point bracket does not contain a root")
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = _collinear_equilibrium(midpoint, mu)
        if abs(f_midpoint) <= tolerance or upper - lower <= tolerance:
            return midpoint
        if f_lower * f_midpoint <= 0.0:
            upper = midpoint
            f_upper = f_midpoint
        else:
            lower = midpoint
            f_lower = f_midpoint
    return 0.5 * (lower + upper)
