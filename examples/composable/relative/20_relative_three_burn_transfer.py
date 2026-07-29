"""Relative composable example 20: CWH three-burn initial design.

The first phase is a natural coast. Transfer 1 begins with the departure
impulse, transfer 2 begins with an optimized intermediate impulse, and the
terminal impulse matches the requested final RIC velocity. The intermediate
burn velocity and timing remain free at a specified RIC waypoint. CWH makes
this broad design-space search fast; switch the shared dynamics to
``Dynamics.relative(...)`` to refine the same phase structure with exact
nonlinear propagation.
"""

from __future__ import annotations

import numpy as np

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    constraints,
    links,
    objectives,
    state,
    variables,
)
from octavian.solvers import SolverOptions

CHIEF_RADIUS_M = EARTH.mean_radius_m + 400_000.0
chief_eci = state(
    [CHIEF_RADIUS_M, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / CHIEF_RADIUS_M), 0.0],
)
initial_ric = state([0.0, -1_000.0, 100.0], [0.0, 0.0, 0.0])
midpoint_seed_ric = state([100.0, -500.0, 50.0], [0.0, 0.0, 0.0])
target_ric = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])
deputy = Spacecraft(name="Deputy", dry_mass_kg=250.0)
dynamics = Dynamics.cwh(
    chief_orbit_radius_m=CHIEF_RADIUS_M,
    chief_initial_state_eci=chief_eci,
)

initial_coast = Phase(
    name="initial_coast",
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    initial_state=initial_ric,
    tof_bounds_s=(120.0, 600.0),
    tof_is_relative=True,
    constraints=[constraints.state(initial_ric, where="Front")],
)
transfer_1 = Phase(
    name="transfer_1",
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    previous=initial_coast,
    link=links.impulsive(),
    final_state=midpoint_seed_ric,
    tof_bounds_s=(500.0, 800.0),
    tof_is_relative=True,
    variables=[variables.impulsive_delta_v(at="Front")],
)
transfer_2 = Phase(
    name="transfer_2",
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    previous=transfer_1,
    link=links.impulsive(),
    final_state=target_ric,
    tof_bounds_s=(500.0, 800.0),
    tof_is_relative=True,
    constraints=[
        constraints.position(midpoint_seed_ric.r_m, where="Front"),
        constraints.state(target_ric, where="Back"),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)

mission = Mission(
    name="Composable: CWH relative three-burn initial design",
    phases=[initial_coast, transfer_1, transfer_2],
    objectives=[objectives.minimize_total_delta_v()],
    mesh_nsegs_precoast=16,
    mesh_nsegs_transfer=24,
    lambert_grid_size=24,
    solver_options=SolverOptions(print_level=0),
)

solution = mission.solve()
print(solution.summary())
solution.viz().save_html(
    "traj_composable_relative_three_burn.html",
    title=mission.name,
)
solution.viz().save_diagnostics_html(
    "diagnostics_composable_relative_three_burn.html",
    title="Three-burn relative state history",
)
