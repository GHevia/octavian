"""User-facing constraint declarations for composable missions.

The constraint objects in this module are intentionally small, explicit, and
Pythonic. They describe *what* the user wants the mission to satisfy; solver
backends decide *how* to compile them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from .specs import BoundaryState

Where = Literal["Front", "Back", "Path"]


def _normalize_where(where: str) -> Where:
    """Normalize user spelling variants for constraint locations."""
    normalized = (where or "").strip().lower()
    if normalized in ("front", "start", "initial", "t0"):
        return "Front"
    if normalized in ("back", "end", "final", "tf"):
        return "Back"
    if normalized in ("path", "all", "trajectory"):
        return "Path"
    raise ValueError(f"Unknown where={where!r}. Use 'front', 'back', or 'path'.")


def _finite_float(name: str, value: float) -> float:
    """Validate and return a finite scalar float."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _optional_nonnegative_float(name: str, value: float | None) -> float | None:
    """Validate an optional non-negative scalar float."""
    if value is None:
        return None
    scalar = _finite_float(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return scalar


class Constraint(ABC):
    """Abstract base class for all mission constraints."""

    kind: ClassVar[str]
    family: ClassVar[str]
    where: Where

    @property
    @abstractmethod
    def value(self) -> Any:
        """Return the canonical constraint payload."""


class OrbitalElementConstraint(Constraint):
    """Marker base class for orbital-element constraints."""

    family: ClassVar[str] = "orbital_element"

    @property
    @abstractmethod
    def element_name(self) -> str:
        """Human-readable orbital element name."""


@dataclass(frozen=True, slots=True)
class SemiMajorAxis(OrbitalElementConstraint):
    """Constrain semi-major axis in meters."""

    kind: ClassVar[str] = "semi_major_axis"

    a_m: float = 0.0
    where: Where = "Path"
    tol_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        semi_major_axis_m = _finite_float("a_m", self.a_m)
        if semi_major_axis_m <= 0.0:
            raise ValueError("a_m must be > 0.")
        object.__setattr__(self, "a_m", semi_major_axis_m)
        tolerance_m = _optional_nonnegative_float("tol_m", self.tol_m)
        if tolerance_m is not None and tolerance_m >= semi_major_axis_m:
            raise ValueError("tol_m must be smaller than a_m.")
        object.__setattr__(self, "tol_m", tolerance_m)

    @property
    def element_name(self) -> str:
        return "semi_major_axis"

    @property
    def value(self) -> dict[str, float | None]:
        return {"a_m": float(self.a_m), "tol_m": self.tol_m}


@dataclass(frozen=True, slots=True)
class Eccentricity(OrbitalElementConstraint):
    """Constrain orbital eccentricity.

    Circular targets are intentionally rejected for now because they need a
    non-singular formulation.
    """

    kind: ClassVar[str] = "eccentricity"

    e: float = 0.0
    where: Where = "Path"
    tol: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        eccentricity_value = _finite_float("e", self.e)
        if not (0.0 < eccentricity_value < 1.0):
            raise ValueError(
                "Eccentricity constraint currently requires 0 < e < 1. "
                "Circular and non-elliptic handling is deferred."
            )
        object.__setattr__(self, "e", eccentricity_value)
        tolerance = _optional_nonnegative_float("tol", self.tol)
        if tolerance is not None and tolerance >= eccentricity_value:
            raise ValueError("tol must be smaller than e for eccentricity constraints.")
        object.__setattr__(self, "tol", tolerance)

    @property
    def element_name(self) -> str:
        return "eccentricity"

    @property
    def value(self) -> dict[str, float | None]:
        return {"e": float(self.e), "tol": self.tol}


@dataclass(frozen=True, slots=True)
class InclinationDeg(OrbitalElementConstraint):
    """Constrain orbital inclination in degrees.

    Equatorial targets are intentionally rejected for now because they need a
    non-singular formulation.
    """

    kind: ClassVar[str] = "inclination_deg"

    inc_deg: float = 0.0
    where: Where = "Path"
    tol_deg: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        inclination_deg = _finite_float("inc_deg", self.inc_deg)
        if not (0.0 < inclination_deg < 180.0):
            raise ValueError(
                "Inclination constraint currently requires 0 < inc_deg < 180. "
                "Equatorial handling is deferred."
            )
        object.__setattr__(self, "inc_deg", inclination_deg)
        tolerance_deg = _optional_nonnegative_float("tol_deg", self.tol_deg)
        if tolerance_deg is not None and tolerance_deg >= min(
            inclination_deg, 180.0 - inclination_deg
        ):
            raise ValueError("tol_deg is too large for the requested inclination target.")
        object.__setattr__(self, "tol_deg", tolerance_deg)

    @property
    def element_name(self) -> str:
        return "inclination_deg"

    @property
    def value(self) -> dict[str, float | None]:
        return {"inc_deg": float(self.inc_deg), "tol_deg": self.tol_deg}


@dataclass(frozen=True, slots=True)
class MinRadius(Constraint):
    """Constrain the trajectory radius norm from below."""

    kind: ClassVar[str] = "min_radius"
    family: ClassVar[str] = "path_geometry"

    r_min_m: float = 0.0
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))

    @property
    def value(self) -> float:
        return float(self.r_min_m)


def _unit_vector(name: str, value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    return vector / magnitude


def _finite_vec3(name: str, value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector


@dataclass(frozen=True, slots=True)
class KeepOutSphere(Constraint):
    """Keep Cartesian position outside a sphere centered in the phase frame."""

    kind: ClassVar[str] = "keep_out_sphere"
    family: ClassVar[str] = "relative_geometry"

    radius_m: float = 0.0
    center_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        radius = _finite_float("radius_m", self.radius_m)
        if radius <= 0.0:
            raise ValueError("radius_m must be > 0")
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "center_m", _finite_vec3("center_m", self.center_m))

    @property
    def value(self) -> dict[str, Any]:
        return {"radius_m": self.radius_m, "center_m": self.center_m}


@dataclass(frozen=True, slots=True)
class ApproachCone(Constraint):
    """Keep position inside a one-sided cone extending from a vertex."""

    kind: ClassVar[str] = "approach_cone"
    family: ClassVar[str] = "relative_geometry"

    axis: Sequence[float] = (1.0, 0.0, 0.0)
    half_angle_deg: float = 30.0
    vertex_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        half_angle = _finite_float("half_angle_deg", self.half_angle_deg)
        if not (0.0 < half_angle < 90.0):
            raise ValueError("half_angle_deg must satisfy 0 < angle < 90")
        object.__setattr__(self, "half_angle_deg", half_angle)
        object.__setattr__(self, "axis", _unit_vector("axis", self.axis))
        object.__setattr__(self, "vertex_m", _finite_vec3("vertex_m", self.vertex_m))

    @property
    def value(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "half_angle_deg": self.half_angle_deg,
            "vertex_m": self.vertex_m,
        }


@dataclass(frozen=True, slots=True)
class LightingAngle(Constraint):
    """Bound the angle from position to a fixed illumination direction.

    ``sun_direction`` is expressed in the same frame as the phase position and
    points from the constraint origin toward the light source.
    """

    kind: ClassVar[str] = "lighting_angle"
    family: ClassVar[str] = "relative_geometry"

    sun_direction: Sequence[float] = (1.0, 0.0, 0.0)
    min_angle_deg: float = 0.0
    max_angle_deg: float = 180.0
    origin_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        minimum = _finite_float("min_angle_deg", self.min_angle_deg)
        maximum = _finite_float("max_angle_deg", self.max_angle_deg)
        if not (0.0 <= minimum < maximum <= 180.0):
            raise ValueError("lighting angles must satisfy 0 <= min < max <= 180")
        object.__setattr__(self, "min_angle_deg", minimum)
        object.__setattr__(self, "max_angle_deg", maximum)
        object.__setattr__(
            self,
            "sun_direction",
            _unit_vector("sun_direction", self.sun_direction),
        )
        object.__setattr__(self, "origin_m", _finite_vec3("origin_m", self.origin_m))

    @property
    def value(self) -> dict[str, Any]:
        return {
            "sun_direction": self.sun_direction,
            "min_angle_deg": self.min_angle_deg,
            "max_angle_deg": self.max_angle_deg,
            "origin_m": self.origin_m,
        }


@dataclass(frozen=True, slots=True)
class SolarPhaseAngle(Constraint):
    """Bound the relative position angle to the ephemeris Sun direction.

    Unlike :class:`LightingAngle`, the direction is not stored as a fixed
    vector.  The composable relative-motion compiler samples the bundled SPICE
    BSP at ``Mission.initial_epoch`` and rotates the Sun line into the chief's
    RIC frame throughout the phase.  CWH dynamics must therefore provide a
    chief initial inertial state.
    """

    kind: ClassVar[str] = "solar_phase_angle"
    family: ClassVar[str] = "relative_geometry"

    min_angle_deg: float = 0.0
    max_angle_deg: float = 180.0
    origin_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        minimum = _finite_float("min_angle_deg", self.min_angle_deg)
        maximum = _finite_float("max_angle_deg", self.max_angle_deg)
        if not (0.0 <= minimum < maximum <= 180.0):
            raise ValueError("solar phase angles must satisfy 0 <= min < max <= 180")
        object.__setattr__(self, "min_angle_deg", minimum)
        object.__setattr__(self, "max_angle_deg", maximum)
        object.__setattr__(self, "origin_m", _finite_vec3("origin_m", self.origin_m))

    @property
    def value(self) -> dict[str, Any]:
        return {
            "min_angle_deg": self.min_angle_deg,
            "max_angle_deg": self.max_angle_deg,
            "origin_m": self.origin_m,
        }


@dataclass(frozen=True, slots=True)
class State(Constraint):
    """Constrain a Cartesian boundary state."""

    kind: ClassVar[str] = "state"
    family: ClassVar[str] = "boundary_state"

    x: BoundaryState = BoundaryState(np.zeros(3), np.zeros(3))
    where: Where = "Front"
    groups: tuple[str, ...] = ("R", "V")

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))

    @property
    def value(self) -> dict[str, Any]:
        return {"x": self.x, "groups": tuple(str(group) for group in self.groups)}


@dataclass(frozen=True, slots=True)
class Position(Constraint):
    """Constrain a Cartesian boundary position."""

    kind: ClassVar[str] = "position"
    family: ClassVar[str] = "boundary_state"

    r_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Front"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))

    @property
    def value(self) -> np.ndarray:
        return np.asarray(self.r_m, dtype=float).reshape(3)


def state(x: BoundaryState, where: str = "Front", groups: Sequence[str] = ("R", "V")) -> State:
    """Create a boundary state constraint."""
    return State(x=x, where=where, groups=tuple(str(group) for group in groups))


def position(r_m: Sequence[float], where: str = "Front") -> Position:
    """Create a boundary position constraint."""
    return Position(r_m=r_m, where=where)


def semi_major_axis(
    a_m: float, where: str = "Path", tol_m: float | None = None
) -> SemiMajorAxis:
    """Create a semi-major-axis constraint."""
    return SemiMajorAxis(a_m=a_m, where=where, tol_m=tol_m)


def eccentricity(e: float, where: str = "Path", tol: float | None = None) -> Eccentricity:
    """Create an eccentricity constraint."""
    return Eccentricity(e=e, where=where, tol=tol)


def inclination_deg(
    inc_deg: float, where: str = "Path", tol_deg: float | None = None
) -> InclinationDeg:
    """Create an inclination constraint."""
    return InclinationDeg(inc_deg=inc_deg, where=where, tol_deg=tol_deg)


def min_radius(r_min_m: float, where: str = "Path") -> MinRadius:
    """Create a minimum-radius constraint."""
    return MinRadius(r_min_m=r_min_m, where=where)


def keep_out_sphere(
    radius_m: float,
    *,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    where: str = "Path",
) -> KeepOutSphere:
    """Create an offset spherical keep-out-zone constraint."""
    return KeepOutSphere(radius_m=radius_m, center_m=center_m, where=where)


def approach_cone(
    axis: Sequence[float],
    half_angle_deg: float,
    *,
    vertex_m: Sequence[float] = (0.0, 0.0, 0.0),
    where: str = "Path",
) -> ApproachCone:
    """Create a one-sided conical approach-corridor constraint."""
    return ApproachCone(
        axis=axis,
        half_angle_deg=half_angle_deg,
        vertex_m=vertex_m,
        where=where,
    )


def lighting_angle(
    sun_direction: Sequence[float],
    *,
    min_angle_deg: float = 0.0,
    max_angle_deg: float = 180.0,
    origin_m: Sequence[float] = (0.0, 0.0, 0.0),
    where: str = "Path",
) -> LightingAngle:
    """Create a fixed-direction lighting-angle constraint."""
    return LightingAngle(
        sun_direction=sun_direction,
        min_angle_deg=min_angle_deg,
        max_angle_deg=max_angle_deg,
        origin_m=origin_m,
        where=where,
    )


def solar_phase_angle(
    *,
    min_angle_deg: float = 0.0,
    max_angle_deg: float = 180.0,
    origin_m: Sequence[float] = (0.0, 0.0, 0.0),
    where: str = "Path",
) -> SolarPhaseAngle:
    """Create an ephemeris-driven RIC solar-phase-angle constraint."""
    return SolarPhaseAngle(
        min_angle_deg=min_angle_deg,
        max_angle_deg=max_angle_deg,
        origin_m=origin_m,
        where=where,
    )
