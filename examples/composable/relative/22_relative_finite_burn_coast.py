"""Relative composable example 22: exact finite burns separated by a coast.

The chief remains unpowered. The deputy executes a short departure burn,
coasts for five minutes, and executes an arrival trim burn. All public states
and plots remain in RIC even though the powered dynamics propagate both
spacecraft in ECI.

Run:
  python examples/composable/relative/22_relative_finite_burn_coast.py
"""

from __future__ import annotations

import numpy as np

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    Thruster,
    constraints,
    objectives,
    state,
)
from octavian.relative import propagate_relative_numerical
from octavian.solvers import SolverOptions

CHIEF_ALTITUDE_M = 400_000.0
TARGET_TIME_S = 420.0


chief_radius_m = EARTH.mean_radius_m + CHIEF_ALTITUDE_M
chief_eci = state(
    [chief_radius_m, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / chief_radius_m), 0.0],
)
initial_ric = state(
    [0.0, -1_000.0, 0.0],
    [0.0, 0.0, 0.0],
)

# Start from the ballistic endpoint, then request a five-meter radial trim.
# This produces a compact example with a real, nonzero finite-burn solution.
ballistic = propagate_relative_numerical(
    chief_eci,
    initial_ric,
    [0.0, TARGET_TIME_S],
    max_step_s=5.0,
)
target_vector = ballistic.relative_states_ric[-1].copy()
target_vector[0] += 5.0
target_ric = state(target_vector[0:3], target_vector[3:6])

deputy = Spacecraft(
    name="Finite-burn deputy",
    dry_mass_kg=250.0,
    thrusters=[
        Thruster(
            name="main",
            thrust_N=10.0,
            isp_s=300.0,
            propellant_mass_kg=10.0,
        )
    ],
)
dynamics = Dynamics.relative(
    chief_initial_state_eci=chief_eci,
    propagation_mode="coupled_eci",
)

departure_burn = Phase(
    name="departure_burn",
    mode="finite_thrust",
    spacecraft=deputy,
    dynamics=dynamics,
    initial_state=initial_ric,
    tof_bounds_s=(59.0, 61.0),
    constraints=[constraints.state(initial_ric, where="Front")],
)
coast = Phase(
    name="relative_coast",
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    previous=departure_burn,
    tof_bounds_s=(299.0, 301.0),
    tof_is_relative=True,
)
arrival_burn = Phase(
    name="arrival_burn",
    mode="finite_thrust",
    spacecraft=deputy,
    dynamics=dynamics,
    previous=coast,
    final_state=target_ric,
    tof_bounds_s=(59.0, 61.0),
    tof_is_relative=True,
    constraints=[constraints.state(target_ric, where="Back")],
)

mission = Mission(
    name="Composable: relative finite-burn coast sequence",
    phases=[departure_burn, coast, arrival_burn],
    objectives=[objectives.minimize_propellant()],
    mesh_nsegs_precoast=8,
    mesh_nsegs_transfer=12,
    lambert_grid_size=12,
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=5,
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The relative finite-burn mission did not return a result.")

print(solution.result.summary())
for powered_phase in solution.result.info["powered_phases"]:
    print(
        f"{powered_phase['phase']}: "
        f"propellant={powered_phase['propellant_used_kg']:.6f} kg, "
        f"equivalent dv={powered_phase['equivalent_dv_mps']:.6f} m/s"
    )

solution.viz().save_html(
    "traj_composable_relative_finite_burn_coast.html",
    title=mission.name,
)
solution.viz().save_diagnostics_html(
    "diagnostics_composable_relative_finite_burn_coast.html",
    title="Relative finite-burn and coast diagnostics",
)
print("Wrote: traj_composable_relative_finite_burn_coast.html")
