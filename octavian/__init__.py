"""Octavian: trajectory optimization and astrodynamics in Python using ASSET."""

# Public, config-like objects
from . import constraints
from . import events
from . import links
from . import objectives
from . import variables
from .conops import rendezvous_two_impulse, rendezvous_precoast_then_transfer
from .events import BoundaryEvent
from .links import Link
from .mission import Mission
from .models import Dynamics, RetryPolicy, RunPlan, SolveConfig
from .objectives import Objective
from .phase import Phase
from .quick import state, two_burn_rendezvous
from .solution import Solution
from .solvers import SolverOptions
from .solvers.rendezvous import RendezvousResult, solve, solve_two_impulse_free_time, solve_two_impulse_precoast
from .spacecraft import Spacecraft, Thruster
from .specs import BoundaryState, TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec
from .study import best_by as study_best_by, grid as study_grid
from .time import cumulative_time_bounds, normalize_time_bounds
from .types import Maneuver

__all__ = [
    # config-like API
    "Thruster",
    "Spacecraft",
    "Dynamics",
    "SolveConfig",
    "RunPlan",
    "RetryPolicy",
    "Phase",
    "Mission",
    "Solution",
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
    "variables",
]
