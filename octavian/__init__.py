"""Octavian: trajectory optimization and astrodynamics in Python using ASSET."""

# Public, config-like objects
from .spacecraft import Spacecraft, Thruster
from .models import Dynamics, SolveConfig, RunPlan, RetryPolicy
from .phase import Phase
from .mission import Mission
from .solution import Solution
from .links import Link
from . import links
from .events import BoundaryEvent
from . import events
from .objectives import Objective
from . import objectives
from . import constraints
from . import variables


# Current v0.x rendezvous specs + solvers (kept for power users)
from .specs import BoundaryState, TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec
from .types import Maneuver
from .solvers import SolverOptions
from .solvers.rendezvous import RendezvousResult, solve, solve_two_impulse_free_time, solve_two_impulse_precoast

# Convenience layers
from .quick import two_burn_rendezvous, state
from .conops import rendezvous_two_impulse, rendezvous_precoast_then_transfer

from .study import grid as study_grid, best_by as study_best_by

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


