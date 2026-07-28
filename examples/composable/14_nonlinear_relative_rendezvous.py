"""Composable example 14: full nonlinear relative rendezvous.

CWH still supplies a fast initial guess, but the optimized phase propagates
chief and deputy absolute states under the exact central-gravity equations.
Inputs, constraints, maneuvers, results, and plots remain in RIC.
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
from octavian.solvers import SolverOptions

chief_radius_m = EARTH.mean_radius_m + 400_000.0
chief_initial_state_eci = state(
    [chief_radius_m, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / chief_radius_m), 0.0],
)
initial_relative_state = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
final_relative_state = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])

phase = Phase(
    name="nonlinear_relative_rendezvous",
    mode="relative_coast",
    spacecraft=Spacecraft(name="Deputy", dry_mass_kg=250.0),
    dynamics=Dynamics.relative(
        chief_initial_state_eci=chief_initial_state_eci,
        chief_name="Chief",
        reference_length_m=1_000.0,
    ),
    initial_state=initial_relative_state,
    final_state=final_relative_state,
    tof_bounds_s=(1_200.0, 2_400.0),
    constraints=[
        constraints.state(initial_relative_state, where="Front"),
        constraints.state(final_relative_state, where="Back"),
        constraints.keep_out_sphere(radius_m=75.0),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)

mission = Mission(
    name="Composable: nonlinear relative rendezvous",
    phases=[phase],
    objectives=[objectives.minimize_total_delta_v()],
    mesh_nsegs_transfer=50,
    lambert_grid_size=80,
    solver_options=SolverOptions(print_level=0),
)

solution = mission.solve()
print(solution.summary())
print(f"Dynamics model: {solution.result.info['dynamics_model']}")
print(
    "Internal state layout: "
    f"{solution.result.info['state_layouts'][0]} "
    "(public result remains [RIC state, time])"
)
print(f"Chief absolute history shape: {solution.chief_trajectory_eci.shape}")
print(f"Deputy absolute history shape: {solution.deputy_trajectory_eci.shape}")

solution.viz().save_html(
    "traj_composable_nonlinear_relative_rendezvous.html",
    title="Full nonlinear relative rendezvous",
)
solution.viz().save_diagnostics_html(
    "diagnostics_composable_nonlinear_relative_rendezvous.html",
    title="Nonlinear relative state over time",
)
