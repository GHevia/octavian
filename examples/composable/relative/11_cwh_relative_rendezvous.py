"""Relative composable example 11: impulsive rendezvous in chief-centered LVLH.

The deputy begins one kilometer behind a chief in a 400 km circular Earth
orbit. CWH dynamics optimize departure and arrival impulses while the solver
chooses a rendezvous time inside the requested window.
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
    objectives,
    state,
    variables,
)
from octavian.relative import cwh_rendezvous_velocity, propagate_cwh
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_relative_trajectory_html

CHIEF_ORBIT_RADIUS_M = EARTH.mean_radius_m + 400_000.0

deputy = Spacecraft(name="Deputy", dry_mass_kg=250.0)
dynamics = Dynamics.cwh(
    chief_orbit_radius_m=CHIEF_ORBIT_RADIUS_M,
    central_body=EARTH,
    chief_name="Chief",
    reference_length_m=1_000.0,
)

initial_relative_state = state(
    r_m=[0.0, -1_000.0, 0.0],
    v_mps=[0.0, 0.0, 0.0],
)
final_relative_state = state(
    r_m=[0.0, 0.0, 0.0],
    v_mps=[0.0, 0.0, 0.0],
)

# The same analytical CWH tools used by Octavian's seed search are available
# for quick feasibility checks and custom initial guesses.
nominal_time_s = 1_800.0
analytic_departure_velocity = cwh_rendezvous_velocity(
    initial_relative_state.r_m,
    final_relative_state.r_m,
    nominal_time_s,
    dynamics.model.mean_motion_radps,
)
analytic_arrival = propagate_cwh(
    np.hstack([initial_relative_state.r_m, analytic_departure_velocity]),
    nominal_time_s,
    dynamics.model.mean_motion_radps,
)

rendezvous = Phase(
    name="relative_rendezvous",
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    initial_state=initial_relative_state,
    final_state=final_relative_state,
    tof_bounds_s=(1_200.0, 2_400.0),
    constraints=[
        constraints.state(initial_relative_state, where="Front"),
        constraints.state(final_relative_state, where="Back"),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)

mission = Mission(
    name="Composable: CWH relative rendezvous",
    phases=[rendezvous],
    objectives=[objectives.minimize_total_delta_v()],
    solver_options=SolverOptions(print_level=0),
    mesh_nsegs_transfer=50,
    lambert_grid_size=80,
)

solution = mission.solve()
print(solution.summary())
print(f"Frame: {solution.frame}")
print(f"Dynamics model: {solution.result.info['dynamics_model']}")
print(f"Analytical CWH departure velocity: {analytic_departure_velocity} m/s")
print(f"Analytical CWH terminal position error: {np.linalg.norm(analytic_arrival[0:3]):.3e} m")

save_relative_trajectory_html(
    solution.traj,
    "traj_composable_cwh_relative_rendezvous.html",
    maneuvers=solution.result.maneuvers,
    phase_segments=solution.result.info["phase_segments"],
    title="CWH rendezvous in the chief RIC frame",
)
solution.viz().save_diagnostics_html(
    "diagnostics_composable_cwh_relative_rendezvous.html",
    title="CWH relative state over time",
)
