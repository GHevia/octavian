"""Boundary-event declarations for phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

BoundaryWhere = Literal["Front", "Back"]


def _normalize_boundary_where(where: str) -> BoundaryWhere:
    """Normalize user spelling variants for boundary locations."""
    normalized = (where or "").strip().lower()
    if normalized in ("front", "start", "initial", "t0"):
        return "Front"
    if normalized in ("back", "end", "final", "tf"):
        return "Back"
    raise ValueError(f"Unknown boundary location {where!r}. Use 'front' or 'back'.")


class BoundaryEvent:
    """Marker base class for boundary events."""

    kind: ClassVar[str]
    where: BoundaryWhere


@dataclass(frozen=True, slots=True)
class Impulse(BoundaryEvent):
    """Declare an impulsive event at a phase boundary."""

    kind: ClassVar[str] = "impulse"
    where: BoundaryWhere = "Front"
    dv_max_mps: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_boundary_where(self.where))


def impulse(where: str, dv_max_mps: float | None = None) -> Impulse:
    """Create an impulsive boundary event.

    Args:
        where: Boundary location, such as ``"Front"`` or ``"Back"``.
        dv_max_mps: Optional delta-v cap metadata for the event.

    Returns:
        An impulse event declaration.
    """
    return Impulse(where=where, dv_max_mps=dv_max_mps)
