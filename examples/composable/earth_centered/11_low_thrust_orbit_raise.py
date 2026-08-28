"""Earth-centered composable example 11: fuel-optimal low-thrust orbit raising.

Run:
  python examples/composable/earth_centered/11_low_thrust_orbit_raise.py
"""

from __future__ import annotations

import numpy as np

from octavian import (
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    Thruster,
    constraints,
    guesses,
    objectives,
    state,
)
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
INITIAL_RADIUS_M = 7_000_000.0
TARGET_RADIUS_M = 8_000_000.0


spacecraft = Spacecraft(
    name="Electric orbit-raising vehicle",
    dry_mass_kg=500.0,
    thrusters=[
        Thruster(
            name="main",
            thrust_N=5.0,
            isp_s=1_800.0,
            propellant_mass_kg=60.0,
        )
    ],
)

initial_state = state(
    [INITIAL_RADIUS_M, 0.0, 0.0],
    [0.0, float(np.sqrt(MU / INITIAL_RADIUS_M)), 0.0],
)
terminal_seed_anchor = state(
    [TARGET_RADIUS_M, 0.0, 0.0],
    [0.0, float(np.sqrt(MU / TARGET_RADIUS_M)), 0.0],
)

orbit_raise = Phase(
    name="electric_orbit_raise",
    mode="low_thrust",
    spacecraft=spacecraft,
    dynamics=Dynamics(mu_m3ps2=MU),
    initial_state=initial_state,
    # This Cartesian state chooses scaling and the spiral target radius. The
    # terminal constraints below leave orbital phase free.
    final_state=terminal_seed_anchor,
    tof_bounds_s=(14.0 * 3_600.0, 24.0 * 3_600.0),
    initial_guess=guesses.low_thrust_spiral(
        throttle=0.85,
        steps_per_orbit=120,
    ),
    constraints=[
        constraints.state(initial_state, where="Front"),
        constraints.min_radius(6_900_000.0),
        constraints.semi_major_axis(TARGET_RADIUS_M, where="Back", tol_m=10_000.0),
        # The current eccentricity declaration requires a strictly positive
        # lower bound, so this expresses a near-circular target band.
        constraints.eccentricity(0.01, where="Back", tol=0.0099),
    ],
)

mission = Mission(
    name="Composable: low-thrust circular orbit raise",
    phases=[orbit_raise],
    objectives=[objectives.minimize_propellant()],
    mesh_nsegs_transfer=100,
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=5,
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)


solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The low-thrust orbit-raising mission did not return a result.")

print(solution.result.summary())
seed_info = solution.result.info["phase_guess_info"][0]
powered = solution.result.info["powered_phases"][0]
print(
    "Spiral seed: "
    f"tof={seed_info['seed_tof_s'] / 3_600.0:.3f} h, "
    f"final radius={seed_info['seed_final_radius_m'] / 1_000.0:.3f} km"
)
print(
    "Optimized propulsion: "
    f"propellant={powered['propellant_used_kg']:.3f} kg, "
    f"equivalent dv={powered['equivalent_dv_mps']:.3f} m/s"
)
for row in solution.result.info["constraint_report"]:
    print(
        f"{row['constraint']}: actual={row['actual']:.6f}, "
        f"target={row['target']:.6f}, ok={row['satisfied']}"
    )

output_path = "traj_composable_low_thrust_orbit_raise.html"
save_trajectory_html(
    solution.result.traj,
    output_path,
    phase_segments=solution.result.info["phase_segments"],
    title=mission.name,
)
solution.viz().save_diagnostics_html(
    "diagnostics_composable_low_thrust_orbit_raise.html",
    title="Low-thrust orbit-raise diagnostics",
)
print(f"Wrote: {output_path}")
