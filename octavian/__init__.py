"""Octavian: trajectory optimization and astrodynamics in Python using ASSET."""

from .specs import BoundaryState, TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec
from .types import Maneuver
from .solvers.rendezvous import (
    RendezvousResult,
    solve,
    solve_two_impulse_free_time,
    solve_two_impulse_precoast,
)

__all__ = [
    "BoundaryState",
    "TwoImpulseFreeTimeSpec",
    "TwoImpulsePreCoastSpec",
    "Maneuver",
    "RendezvousResult",
    "solve",
    "solve_two_impulse_free_time",
    "solve_two_impulse_precoast",
]
