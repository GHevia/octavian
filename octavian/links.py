"""Phase-linking semantics for composable missions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Link:
    """Describe continuity requirements across a phase boundary."""

    kind: str = "continuous"
    dv_max_mps: float | None = None
    name: str = "link"

    def is_impulsive(self) -> bool:
        """Return whether the link allows a boundary velocity jump."""
        return self.kind.lower() == "impulsive"

    def is_continuous(self) -> bool:
        """Return whether the link enforces full state continuity."""
        return self.kind.lower() == "continuous"


def continuous(*, name: str = "continuous") -> Link:
    """Create a continuous link between phases.

    Args:
        name: Human-readable name for the link object.

    Returns:
        A link that enforces continuity in position, velocity, and time.
    """
    return Link(kind="continuous", name=name)


def impulsive(*, dv_max_mps: float | None = None, name: str = "impulsive") -> Link:
    """Create an impulsive link between phases.

    Args:
        dv_max_mps: Optional delta-v cap metadata for the link.
        name: Human-readable name for the link object.

    Returns:
        A link that enforces continuity in position and time while allowing a
        velocity jump at the boundary.

    Notes:
        In v0.x solvers, delta-v is represented implicitly by allowing
        front/back boundary velocities to differ and optionally adding a
        delta-v objective term. If ``dv_max_mps`` is provided, current solvers
        treat it as soft metadata unless a solver adds an explicit hard bound.
    """
    return Link(kind="impulsive", dv_max_mps=dv_max_mps, name=name)
