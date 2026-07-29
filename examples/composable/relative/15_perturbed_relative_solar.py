"""Relative composable example 15: perturbed rendezvous with a SPICE Sun constraint.

Chief and deputy absolute states are propagated under central gravity, J2, and
solar gravity. The public constraints and results remain in RIC. The
solar-phase angle uses the bundled BSP and mission epoch.
"""

from __future__ import annotations

import numpy as np

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Perturbations,
    Phase,
    Spacecraft,
    constraints,
    objectives,
    state,
    variables,
)
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_relative_trajectory_html

CHIEF_ORBIT_RADIUS_M = EARTH.mean_radius_m + 400_000.0
CHIEF_ORBIT_SPEED_MPS = np.sqrt(EARTH.mu_m3ps2 / CHIEF_ORBIT_RADIUS_M)

chief_initial_state_eci = state(
    [CHIEF_ORBIT_RADIUS_M, 0.0, 0.0],
    [0.0, CHIEF_ORBIT_SPEED_MPS, 0.0],
)
initial_relative_state = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
stand_off_state = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])

dynamics = Dynamics.relative(
    chief_name="Chief",
    chief_initial_state_eci=chief_initial_state_eci,
    perturbations=Perturbations(j2=True, sun=True),
    third_body_table_step_s=300.0,
)

approach = Phase(
    name="perturbed_sun_safe_approach",
    mode="relative_coast",
    spacecraft=Spacecraft(name="Deputy", dry_mass_kg=250.0),
    dynamics=dynamics,
    initial_state=initial_relative_state,
    final_state=stand_off_state,
    tof_bounds_s=(1_200.0, 2_400.0),
    constraints=[
        constraints.state(initial_relative_state, where="Front"),
        constraints.state(stand_off_state, where="Back"),
        constraints.keep_out_sphere(radius_m=75.0),
        constraints.solar_phase_angle(
            min_angle_deg=20.0,
            max_angle_deg=145.0,
        ),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)

mission = Mission(
    name="Composable: perturbed relative solar geometry",
    initial_epoch="2026-01-01T00:00:00Z",
    phases=[approach],
    objectives=[objectives.minimize_total_delta_v()],
    solver_options=SolverOptions(print_level=0),
    mesh_nsegs_transfer=60,
    lambert_grid_size=80,
)

solution = mission.solve()
print(solution.summary())
print(f"Dynamics model: {solution.result.info['dynamics_model']}")
print(f"Constraint report: {solution.result.info['constraint_report']}")

save_relative_trajectory_html(
    solution.traj,
    "traj_composable_perturbed_relative_solar.html",
    maneuvers=solution.result.maneuvers,
    phase_segments=solution.result.info["phase_segments"],
    chief_radius_m=75.0,
    title="Perturbed relative approach with SPICE solar geometry",
)
solution.viz().save_diagnostics_html(
    "diagnostics_composable_perturbed_relative_solar.html",
    title="Perturbed relative state and solar geometry",
)
