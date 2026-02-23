# octavian/constraints.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from .specs import BoundaryState

Where = Literal["Front", "Back", "Path"]


def _norm_where(where: str) -> Where:
    w = (where or "").strip().lower()
    if w in ("front", "start", "initial", "t0"):
        return "Front"
    if w in ("back", "end", "final", "tf"):
        return "Back"
    if w in ("path", "all", "trajectory"):
        return "Path"
    raise ValueError(f"Unknown where={where!r}. Use 'front', 'back', or 'path'.")


class Constraint(ABC):
    """Base class for constraints with a uniform value accessor."""

    kind: ClassVar[str]
    where: Where

    @property
    @abstractmethod
    def value(self) -> Any:
        """Canonical payload for this constraint."""


@dataclass(frozen=True, slots=True)
class SemiMajorAxis(Constraint):
    kind: ClassVar[str] = "semi_major_axis"
    a_m: float = 0.0
    where: Where = "Path"
    tol_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))

    @property
    def value(self) -> dict[str, float | None]:
        return {
            "a_m": float(self.a_m),
            "tol_m": (None if self.tol_m is None else float(self.tol_m)),
        }


@dataclass(frozen=True, slots=True)
class InclinationDeg(Constraint):
    kind: ClassVar[str] = "inclination_deg"
    inc_deg: float = 0.0
    where: Where = "Path"
    tol_deg: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))

    @property
    def value(self) -> dict[str, float | None]:
        return {
            "inc_deg": float(self.inc_deg),
            "tol_deg": (None if self.tol_deg is None else float(self.tol_deg)),
        }


@dataclass(frozen=True, slots=True)
class MinRadius(Constraint):
    """Minimum radius magnitude constraint (path or boundary)."""

    kind: ClassVar[str] = "min_radius"
    r_min_m: float = 0.0
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))

    @property
    def value(self) -> float:
        return float(self.r_min_m)


@dataclass(frozen=True, slots=True)
class State(Constraint):
    """Fix a boundary Cartesian state.

    Default semantics: constrain R and V at the specified boundary.
    The composable ASSET compiler may *relax* the V constraint if an
    ImpulsiveDeltaV variable exists at the same boundary, and instead
    treat the difference to the desired V as an objective term.
    """

    kind: ClassVar[str] = "state"
    x: BoundaryState = BoundaryState(np.zeros(3), np.zeros(3))
    where: Where = "Front"
    # which semantic groups to enforce ("R","V","t")
    groups: tuple[str, ...] = ("R", "V")

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))

    @property
    def value(self) -> dict[str, Any]:
        return {"x": self.x, "groups": tuple(str(g) for g in self.groups)}


@dataclass(frozen=True, slots=True)
class Position(Constraint):
    """Fix a boundary position vector."""

    kind: ClassVar[str] = "position"
    r_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Front"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))

    @property
    def value(self) -> np.ndarray:
        return np.asarray(self.r_m, dtype=float).reshape(3)


# Factory helpers (nice user API)
def state(x: BoundaryState, where: str = "Front", groups: Sequence[str] = ("R", "V")) -> State:
    return State(x=x, where=where, groups=tuple(str(g) for g in groups))


def position(r_m: Sequence[float], where: str = "Front") -> Position:
    return Position(r_m=r_m, where=where)


# Factory helpers (nice user API)
def semi_major_axis(a_m: float, where: str = "Path", tol_m: float | None = None) -> SemiMajorAxis:
    return SemiMajorAxis(a_m=a_m, where=where, tol_m=tol_m)


def inclination_deg(
    inc_deg: float, where: str = "Path", tol_deg: float | None = None
) -> InclinationDeg:
    return InclinationDeg(inc_deg=inc_deg, where=where, tol_deg=tol_deg)


def min_radius(r_min_m: float, where: str = "Path") -> MinRadius:
    return MinRadius(r_min_m=r_min_m, where=where)
