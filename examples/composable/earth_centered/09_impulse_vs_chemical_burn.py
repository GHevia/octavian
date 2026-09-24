"""Earth-centered composable example 09: impulsive versus finite chemical burn.

Minimize propellant for the finite burn-coast-burn mission, then compare it
with a two-impulse Lambert transfer between the same states at the same total
flight time. Burn durations count toward that time; the coast alone does not.

Run:
  python examples/composable/earth_centered/09_impulse_vs_chemical_burn.py
"""

from __future__ import annotations

import numpy as np

from octavian import Mission, Phase, Spacecraft, Thruster, constraints, objectives, state
from octavian.astro import kepler_dense_guess, select_best_lambert_seed
from octavian.models import Dynamics
from octavian.solvers import SolverOptions
from octavian.viz.matplotlib import save_trajectory_image
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
COAST_BOUNDS_S = (1_800.0, 3_000.0)
LAMBERT_GRID_SIZE = 201


initial_state = state(
    r_m=[7000e3, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
)
target_state = state(
    r_m=[-5_951_609.571397256, 3_684_880.3928557364, 0.0],
    v_mps=[-3_972.32911602311, -6_395.880426811104, 0.0],
)
dynamics = Dynamics(mu_m3ps2=MU)


chemical_spacecraft = Spacecraft(
    name="Chemical demo spacecraft",
    dry_mass_kg=500.0,
    thrusters=[
        Thruster(
            name="main",
            thrust_N=2_000.0,
            isp_s=320.0,
            propellant_mass_kg=50.0,
        )
    ],
)

departure_burn = Phase(
    name="departure_burn",
    mode="chemical_burn",
    spacecraft=chemical_spacecraft,
    dynamics=dynamics,
    initial_state=initial_state,
    tof_bounds_s=(20.0, 120.0),
    constraints=[constraints.state(initial_state, where="Front")],
)

coast = Phase(
    name="coast",
    mode="coast",
    spacecraft=chemical_spacecraft,
    previous=departure_burn,
    dynamics=dynamics,
    tof_bounds_s=COAST_BOUNDS_S,
    tof_is_relative=True,
)

arrival_burn = Phase(
    name="arrival_burn",
    mode="chemical_burn",
    spacecraft=chemical_spacecraft,
    previous=coast,
    dynamics=dynamics,
    final_state=target_state,
    tof_bounds_s=(20.0, 120.0),
    tof_is_relative=True,
    constraints=[constraints.state(target_state, where="Back")],
)

chemical_mission = Mission(
    name="Composable: finite chemical burn transfer",
    phases=[departure_burn, coast, arrival_burn],
    # Optimize fuel use before comparing the rocket-equation equivalent
    # delta-v with an impulsive transfer at the same final time.
    objectives=[objectives.minimize_propellant()],
    solver_options=SolverOptions(
        print_level=0, max_ls_iters=5, enable_adaptive_mesh=True, asset_threads=(1, 1)
    ),
    mesh_nsegs_precoast=30,
    mesh_nsegs_transfer=60,
    lambert_grid_size=LAMBERT_GRID_SIZE,
    nrevs_to_try=(0,),
)


chemical_solution = chemical_mission.solve()

if not chemical_solution.ok or chemical_solution.result is None:
    raise RuntimeError("The finite-burn transfer must solve before delta-v can be compared.")

# Compare Lambert branches at the solved total flight time.
seed = select_best_lambert_seed(
    r0_m=initial_state.r_m,
    rf_m=target_state.r_m,
    v0_mps=initial_state.v_mps,
    vf_mps=target_state.v_mps,
    mu_m3ps2=MU,
    tmin_s=chemical_solution.result.tf_s(),
    tmax_s=chemical_solution.result.tf_s(),
    nrevs=(0,),
)
impulsive_traj = np.asarray(
    kepler_dense_guess(
        r0_m=initial_state.r_m,
        v0_mps=seed.v1_mps,
        t0_s=0.0,
        tf_s=seed.tof_s,
        npts=80,
        mu_m3ps2=MU,
    ),
    dtype=float,
)
impulsive_dv_mps = float(seed.total_dv_mps)
chemical_dv_mps = sum(
    float(burn["equivalent_dv_mps"]) for burn in chemical_solution.result.info["chemical_burns"]
)
relative_difference = abs(chemical_dv_mps - impulsive_dv_mps) / max(impulsive_dv_mps, 1.0)

print(chemical_solution.result.summary())
print(f"Shared flight time: {chemical_solution.result.tf_s():.3f} s (including both burns)")
print(
    "Delta-v comparison: "
    f"impulsive={impulsive_dv_mps:.3f} m/s, "
    f"chemical equivalent={chemical_dv_mps:.3f} m/s, "
    f"relative difference={100.0 * relative_difference:.2f}%"
)
if not np.isfinite(relative_difference) or relative_difference > 0.20:
    raise RuntimeError(
        "Finite-burn equivalent delta-v differs from the impulsive reference by more than 20%."
    )

# Compare both trajectories in the same view at their shared flight time.
reference_trajectories = [
    {
        "name": "Two-impulse Lambert reference",
        "traj": impulsive_traj,
        "color": "#F59E0B",
    }
]

impulse_html = "traj_composable_impulse_reference.html"
chemical_html = "traj_composable_chemical_reference.html"
save_trajectory_html(
    impulsive_traj,
    impulse_html,
    title="Two-impulse Lambert reference at the same final time",
)
save_trajectory_html(
    chemical_solution.result.traj,
    chemical_html,
    reference_trajectories=reference_trajectories,
    phase_segments=chemical_solution.result.info.get("phase_segments", []),
    title=chemical_mission.name,
)
print(f"Wrote: {impulse_html}")
print(f"Wrote: {chemical_html}")

save_trajectory_image(
    chemical_solution.traj,
    "traj_composable_chemical_reference.png",
    reference_trajectories=reference_trajectories,
    phase_segments=chemical_solution.result.info.get("phase_segments"),
    projection="xy",
    title="Finite chemical burns and impulsive reference — XY plane",
)
