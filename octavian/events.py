from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Literal

BoundaryWhere = Literal["Front", "Back"]


def _norm_where(where: str) -> BoundaryWhere:
    w = (where or "").strip().lower()
    if w in ("front", "start", "initial", "t0"):
        return "Front"
    if w in ("back", "end", "final", "tf"):
        return "Back"
    raise ValueError(f"Unknown boundary location {where!r}. Use 'front' or 'back'.")


class BoundaryEvent:
    """Marker base class for boundary events."""
    kind: ClassVar[str]
    where: BoundaryWhere


@dataclass(frozen=True, slots=True)
class Impulse(BoundaryEvent):
    """Impulsive Δv at a phase boundary."""
    kind: ClassVar[str] = "impulse"
    where: BoundaryWhere = "Front"
    dv_max_mps: Optional[float] = None

    def __post_init__(self) -> None:
        # normalize aliases like "start", "t0", etc.
        object.__setattr__(self, "where", _norm_where(self.where))


def impulse(where: str, dv_max_mps: Optional[float] = None) -> Impulse:
    return Impulse(where=where, dv_max_mps=dv_max_mps)
