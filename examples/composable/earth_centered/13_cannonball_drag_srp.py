"""Earth-centered example 13: inertial cannonball drag and SRP recapture.

The nominal endpoint comes from a two-body coast. The composable solve then
uses Earth J2, a co-rotating exponential atmosphere, and BSP-driven solar
radiation pressure. Boundary impulses expose the correction needed to recover
the nominal orbit after one quarter revolution.

Run:
  python examples/composable/earth_centered/13_cannonball_drag_srp.py
"""

from __future__ import annotations

import numpy as np

from octavian import (
    EARTH,
    Cannonball,
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
from octavian.astro import estimate_orbital_period_s, propagate_cartesian_rv
from octavian.solvers import SolverOptions

INITIAL_EPOCH = "2026-01-01T00:00:00Z"
orbit_radius_m = EARTH.mean_radius_m + 400_000.0
initial_state = state(
    [orbit_radius_m, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / orbit_radius_m), 0.0],
)
period_s = estimate_orbital_period_s(
    initial_state.r_m,
    initial_state.v_mps,
    EARTH.mu_m3ps2,
)
if period_s is None:
    raise RuntimeError("The demonstration requires a closed elliptic orbit.")
coast_duration_s = 0.25 * period_s
target_vector = propagate_cartesian_rv(
    np.hstack([initial_state.r_m, initial_state.v_mps]),
    coast_duration_s,
    EARTH.mu_m3ps2,
)
target_state = state(target_vector[0:3], target_vector[3:6])

spacecraft = Spacecraft(
    name="Cannonball demonstrator",
    dry_mass_kg=200.0,
    cannonball=Cannonball(
        drag_area_m2=4.0,
        drag_coefficient=2.2,
        srp_area_m2=6.0,
        reflectivity_coefficient=1.4,
    ),
)
dynamics = Dynamics.for_body(
    EARTH,
    perturbations=Perturbations(
        j2=True,
        drag=True,
        srp=True,
    ),
    third_body_table_step_s=600.0,
)

recapture = Phase(
    name="perturbed_orbit_recapture",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    initial_state=initial_state,
    final_state=target_state,
    tof_bounds_s=(coast_duration_s - 10.0, coast_duration_s + 10.0),
    constraints=[
        constraints.state(initial_state, where="Front"),
        constraints.state(target_state, where="Back"),
    ],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)

mission = Mission(
    name="Earth orbit with cannonball drag and SRP",
    initial_epoch=INITIAL_EPOCH,
    phases=[recapture],
    objectives=[objectives.minimize_total_delta_v()],
    mesh_nsegs_transfer=40,
    lambert_grid_size=20,
    nrevs_to_try=(0,),
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=5,
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The inertial cannonball mission did not return a result.")

print(solution.summary())
print(f"Active force model: {dynamics.active_perturbations()}")
print(
    "Quarter-orbit recapture delta-v: "
    f"{sum(np.linalg.norm(item.dv_mps) for item in solution.result.maneuvers):.6f} m/s"
)
solution.viz().save_html(
    "traj_inertial_cannonball_drag_srp.html",
    title=mission.name,
)
solution.viz().save_diagnostics_html(
    "diagnostics_inertial_cannonball_drag_srp.html",
    title=mission.name,
)
