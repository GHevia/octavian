"""High-level configuration models.

These classes exist to keep user scripts readable while still allowing
advanced solving behavior (continuation / retries) via sensible defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Perturbations:
    """Perturbation flags for translational dynamics.

    The composable ASSET backend supports the core Earth-orbit perturbations:
    J2, lunar third-body gravity, and solar third-body gravity. ``moon`` and
    ``sun`` are convenience flags; ``third_bodies=("moon", "sun")`` remains
    accepted for scripts that prefer a body list.
    """

    j2: bool = False
    moon: bool = False
    sun: bool = False
    srp: bool = False
    drag: bool = False
    third_bodies: tuple[str, ...] = ()

    def active_third_bodies(self) -> tuple[str, ...]:
        """Return normalized third-body names requested by this config."""
        names: list[str] = []
        if self.moon:
            names.append("moon")
        if self.sun:
            names.append("sun")
        for body in self.third_bodies:
            normalized = str(body).strip().lower().replace("-", "_")
            if normalized in {"luna", "earth_moon"}:
                normalized = "moon"
            if normalized not in names:
                names.append(normalized)
        return tuple(names)


@dataclass(slots=True)
class Dynamics:
    """Environment and dynamics configuration."""

    mu_m3ps2: float = 3.986004418e14
    central_body_radius_m: float = 6_378_136.3
    j2_coefficient: float = 1.08262668e-3
    third_body_table_step_s: float = 3600.0
    third_body_table_margin_s: float = 86400.0
    third_bodies: tuple[str, ...] = ()
    j2: bool = False
    moon: bool = False
    sun: bool = False
    srp: bool = False
    drag: bool = False
    perturbations: Perturbations | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def active_perturbations(self) -> Perturbations:
        """Return normalized perturbation flags for solver compilation."""
        if self.perturbations is not None:
            return self.perturbations
        return Perturbations(
            j2=bool(self.j2),
            moon=bool(self.moon),
            sun=bool(self.sun),
            srp=bool(self.srp),
            drag=bool(self.drag),
            third_bodies=tuple(str(body) for body in self.third_bodies),
        )


@dataclass(slots=True)
class SolveConfig:
    """Runner behavior settings."""

    max_attempts: int = 3
    raise_on_fail: bool = True
    verbose: bool = True


@dataclass(slots=True)
class Stage:
    """A single continuation stage (minimal in v0.x)."""

    name: str
    nsegs_scale: float | None = None
    tighten_bounds: bool = False


@dataclass(slots=True)
class RunPlan:
    """Continuation / crawl-walk-run plan."""

    stages: Sequence[Stage] = ()

    @staticmethod
    def default() -> RunPlan:
        return RunPlan(stages=())


@dataclass(slots=True)
class RetryPolicy:
    """Retry behavior when a solve fails."""

    enabled: bool = True
    max_retries: int = 2

    @staticmethod
    def default() -> RetryPolicy:
        return RetryPolicy(enabled=True, max_retries=2)
