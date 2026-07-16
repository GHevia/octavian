"""Octavian: trajectory optimization and astrodynamics in Python using ASSET."""

# Public, config-like objects
from . import bodies, constraints, coordinates, events, links, objectives, relative, variables
from .bodies import EARTH, MOON, SUN, CelestialBody
from .conops import rendezvous_precoast_then_transfer, rendezvous_two_impulse
from .coordinates import CoordinateFrame, SolverScaling, StateLayout
from .events import BoundaryEvent
from .links import Link
from .mission import Mission
from .models import Dynamics, Perturbations, RetryPolicy, RunPlan, SolveConfig
from .objectives import Objective
from .phase import Phase
from .quick import state, two_burn_rendezvous
from .relative import ClohessyWiltshire
from .solution import Solution
from .solvers import SolverOptions
from .solvers.preconfigured import (
    RendezvousResult,
    solve,
    solve_two_impulse_free_time,
    solve_two_impulse_precoast,
)
from .spacecraft import Spacecraft, Thruster
from .specs import BoundaryState, TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec
from .study import best_by as study_best_by
from .study import grid as study_grid
from .time import cumulative_time_bounds, normalize_time_bounds
from .types import Maneuver

__all__ = [
    # config-like API
    "Thruster",
    "Spacecraft",
    "Dynamics",
    "Perturbations",
    "SolveConfig",
    "RunPlan",
    "RetryPolicy",
    "Phase",
    "Mission",
    "Solution",
    "CoordinateFrame",
    "SolverScaling",
    "StateLayout",
    "CelestialBody",
    "ClohessyWiltshire",
    "EARTH",
    "MOON",
    "SUN",
    # quick + conops
    "two_burn_rendezvous",
    "state",
    "rendezvous_two_impulse",
    "rendezvous_precoast_then_transfer",
    "cumulative_time_bounds",
    "normalize_time_bounds",
    # rendezvous specs + solver
    "BoundaryState",
    "TwoImpulseFreeTimeSpec",
    "TwoImpulsePreCoastSpec",
    "SolverOptions",
    "RendezvousResult",
    "solve",
    "solve_two_impulse_free_time",
    "solve_two_impulse_precoast",
    "Maneuver",
    # study helpers
    "study_grid",
    "study_best_by",
    # constraints / events / links / objectives
    "Link",
    "BoundaryEvent",
    "Objective",
    "links",
    "events",
    "objectives",
    "constraints",
    "bodies",
    "coordinates",
    "relative",
    "variables",
]
