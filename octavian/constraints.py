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
