"""Octavian: trajectory optimization and astrodynamics in Python using ASSET."""

# ruff: noqa: E402
# Native DLL setup must run before importing the public API below. Several
# solver modules import ASSET while they initialize and cache that availability.
from ._runtime import enable_native_runtime

# Keep the handle alive: Python removes an added DLL directory when its handle
# is collected. This makes the pip+venv and conda installation paths equivalent
# for ASSET's Windows OpenMP runtime without changing the global system PATH.
_NATIVE_RUNTIME_HANDLES = enable_native_runtime()

# Public, config-like objects
from . import (
    bodies,
    config,
    constraints,
    coordinates,
    events,
    forces,
    guesses,
    links,
    objectives,
    relative,
    variables,
)
from .bodies import EARTH, MOON, SUN, CelestialBody
from .config import MissionConfigError, load_mission, load_mission_mapping, mission_from_dict
from .conops import rendezvous_precoast_then_transfer, rendezvous_two_impulse
from .control import ThrustControl
from .coordinates import CoordinateFrame, SolverScaling, StateLayout
from .events import BoundaryEvent
from .exports import Ephemeris
from .forces import (
    EARTH_EXPONENTIAL_ATMOSPHERE,
    Cannonball,
    ExponentialAtmosphere,
    cannonball_drag_acceleration,
    cannonball_srp_acceleration,
)
from .guesses import LowThrustSpiralGuess
from .links import Link
from .mission import Mission
from .models import Dynamics, Perturbations, RetryPolicy, RunPlan, SolveConfig
from .objectives import Objective
from .phase import Phase
from .quick import (
    relative_hop,
    relative_transfer_chain,
    state,
    two_burn_rendezvous,
)
from .relative import ClohessyWiltshire, NonlinearRelative, RelativePropagationMode
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
    "ThrustControl",
    "Spacecraft",
    "Cannonball",
    "ExponentialAtmosphere",
    "EARTH_EXPONENTIAL_ATMOSPHERE",
    "cannonball_drag_acceleration",
    "cannonball_srp_acceleration",
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
    "NonlinearRelative",
    "RelativePropagationMode",
    "LowThrustSpiralGuess",
    "MissionConfigError",
    "EARTH",
    "MOON",
    "SUN",
    "load_mission",
    "load_mission_mapping",
    "mission_from_dict",
    # quick + conops
    "two_burn_rendezvous",
    "relative_hop",
    "relative_transfer_chain",
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
    "Ephemeris",
    "Objective",
    "links",
    "events",
    "forces",
    "guesses",
    "objectives",
    "constraints",
    "bodies",
    "config",
    "coordinates",
    "relative",
    "variables",
]
