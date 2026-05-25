"""Decision-variable declarations for composable missions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

Where = Literal["Front", "Back"]


def _normalize_where(where: str) -> Where:
    """Normalize user spelling variants for boundary locations."""
    normalized = (where or "").strip().lower()
    if normalized in ("front", "start", "initial", "t0"):
        return "Front"
    if normalized in ("back", "end", "final", "tf"):
        return "Back"
    raise ValueError(f"Unknown where={where!r}. Use 'front' or 'back'.")


class Variable:
    """Marker base class for composable decision variables."""

    kind: ClassVar[str]
    where: Where


@dataclass(frozen=True, slots=True)
class ImpulsiveDeltaV(Variable):
    """Declare an impulsive delta-v at a phase boundary.

    The composable compiler interprets this variable declaration to free the
    boundary velocity and add the appropriate objective terms.
    """

    kind: ClassVar[str] = "impulsive_delta_v"
    where: Where = "Front"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))


def impulsive_delta_v(*, at: str = "Front") -> ImpulsiveDeltaV:
    """Create an impulsive delta-v declaration.

    Args:
        at: Boundary location, such as ``"Front"`` or ``"Back"``.

    Returns:
        An impulsive delta-v variable declaration.
    """
    return ImpulsiveDeltaV(where=at)
