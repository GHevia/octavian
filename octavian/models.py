"""High-level configuration models.

These classes exist to keep user scripts readable while still allowing
advanced solving behavior (continuation / retries) via sensible defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Dynamics:
    """Environment and dynamics configuration.

    v0.x note:
      - Only `mu_m3ps2` is actively used by the current rendezvous solvers.
      - Other fields exist to stabilize the API and guide expansion.
    """

    mu_m3ps2: float = 3.986004418e14
    third_bodies: tuple[str, ...] = ()
    j2: bool = False
    srp: bool = False
    drag: bool = False
    info: dict[str, Any] = field(default_factory=dict)


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
