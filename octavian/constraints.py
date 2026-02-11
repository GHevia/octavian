# octavian/constraints.py
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Literal, Sequence


from .specs import BoundaryState
import numpy as np


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


class Constraint:
    """Marker base class for constraints."""
    kind: ClassVar[str]
    where: Where


@dataclass(frozen=True, slots=True)
class SemiMajorAxis(Constraint):
    kind: ClassVar[str] = "semi_major_axis"
    a_m: float = 0.0
    where: Where = "Path"
    tol_m: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))


@dataclass(frozen=True, slots=True)
class InclinationDeg(Constraint):
    kind: ClassVar[str] = "inclination_deg"
    inc_deg: float = 0.0
    where: Where = "Path"
    tol_deg: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))




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


@dataclass(frozen=True, slots=True)
class Position(Constraint):
    """Fix a boundary position vector."""
    kind: ClassVar[str] = "position"
    r_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Front"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))


# Factory helpers (nice user API)
def state(x: BoundaryState, where: str = "Front", groups: Sequence[str] = ("R","V")) -> State:
    return State(x=x, where=where, groups=tuple(str(g) for g in groups))

def position(r_m: Sequence[float], where: str = "Front") -> Position:
    return Position(r_m=r_m, where=where)

# Factory helpers (nice user API)
def semi_major_axis(a_m: float, where: str = "Path", tol_m: Optional[float] = None) -> SemiMajorAxis:
    return SemiMajorAxis(a_m=a_m, where=where, tol_m=tol_m)

def inclination_deg(inc_deg: float, where: str = "Path", tol_deg: Optional[float] = None) -> InclinationDeg:
    return InclinationDeg(inc_deg=inc_deg, where=where, tol_deg=tol_deg)
