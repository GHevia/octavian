"""Decision-variable declarations for composable missions.

In Octavian's composable layer, a "variable" is a user-facing declaration that
affects how the backend problem is constructed (what is free, what is fixed),
and often contributes objective terms and maneuver bookkeeping.

v0.x supports a minimal set:
  - ImpulsiveDeltaV at a phase boundary (Front or Back)

Future: control profiles, thrust direction constraints, mass, attitude, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

Where = Literal["Front", "Back"]


def _norm_where(where: str) -> Where:
    w = (where or "").strip().lower()
    if w in ("front", "start", "initial", "t0"):
        return "Front"
    if w in ("back", "end", "final", "tf"):
        return "Back"
    raise ValueError(f"Unknown where={where!r}. Use 'front' or 'back'.")


class Variable:
    """Marker base class for composable decision variables."""

    kind: ClassVar[str]
    where: Where


@dataclass(frozen=True, slots=True)
class ImpulsiveDeltaV(Variable):
    """Declare an impulsive Δv at a phase boundary.

    Semantics (composable compiler):
      - The boundary velocity is treated as *free* (not fixed by State constraints)
      - An objective term is added to penalize the magnitude of the required Δv.

    For where="Front":
      - If phase has a previous and the link does not enforce velocity continuity,
        Δv is measured between previous.Back.V and this.Front.V.
      - If phase has no previous and a State constraint exists at Front,
        Δv is measured between desired initial velocity and this.Front.V.

    For where="Back":
      - If a State constraint exists at Back providing a desired terminal velocity,
        Δv is measured between this.Back.V and the desired V.
    """

    kind: ClassVar[str] = "impulsive_delta_v"
    where: Where = "Front"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _norm_where(self.where))


def impulsive_delta_v(*, at: str = "Front") -> ImpulsiveDeltaV:
    """Factory helper."""
    return ImpulsiveDeltaV(where=at)
