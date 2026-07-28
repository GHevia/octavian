"""Composable example 12: relative rendezvous with safety and lighting geometry."""

from __future__ import annotations

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

dynamics = Dynamics.cwh(
    chief_orbit_radius_m=EARTH.mean_radius_m + 400_000.0,
    chief_name="Chief",
    reference_length_m=1_000.0,
)
initial_state = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
stand_off_state = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])

approach = Phase(
    name="safe_approach",
    mode="relative_coast",
    spacecraft=Spacecraft(name="Deputy", dry_mass_kg=250.0),
    dynamics=dynamics,
    initial_state=initial_state,
    final_state=stand_off_state,
    tof_bounds_s=(1_200.0, 2_400.0),
    constraints=[
        constraints.state(initial_state, where="Front"),
        constraints.state(stand_off_state, where="Back"),
        constraints.keep_out_sphere(radius_m=75.0),
        constraints.approach_cone(axis=[0.0, -1.0, 0.0], half_angle_deg=30.0),
        constraints.lighting_angle(
            sun_direction=[1.0, 0.0, 0.0],
            min_angle_deg=85.0,
            max_angle_deg=121.0,
        ),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)

mission = Mission(
    name="Composable: CWH safety corridor",
    phases=[approach],
    objectives=[objectives.minimize_total_delta_v()],
    mesh_nsegs_transfer=60,
    lambert_grid_size=80,
    solver_options=SolverOptions(print_level=0),
)

solution = mission.solve()
print(solution.summary())
solution.viz().save_html(
    "traj_composable_cwh_safety_corridor.html",
    title="CWH rendezvous safety corridor",
)
solution.viz().save_diagnostics_html(
    "diagnostics_composable_cwh_safety_corridor.html",
    title="CWH safety-corridor state over time",
)
