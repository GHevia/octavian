"""Octavian: trajectory optimization and astrodynamics in Python using ASSET."""

from .specs import BoundaryState, TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec
from .types import Maneuver
from .solvers import SolverOptions
from .solvers.rendezvous import (
    RendezvousResult,
    solve,
    solve_two_impulse_free_time,
    solve_two_impulse_precoast,
)
from .study import grid as study_grid, best_by as study_best_by

__all__ = [
    "BoundaryState",
    "TwoImpulseFreeTimeSpec",
    "TwoImpulsePreCoastSpec",
    "Maneuver",
    "SolverOptions",
    "RendezvousResult",
    "solve",
    "solve_two_impulse_free_time",
    "solve_two_impulse_precoast",
    "study_grid",
    "study_best_by",
]
