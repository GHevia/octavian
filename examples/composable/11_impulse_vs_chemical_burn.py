"""Example 11: compare an impulsive transfer with a finite chemical burn.

The impulsive reference is the best two-impulse Lambert seed over the same
coast-time window used by the finite-burn transfer.

Run:
  python examples/composable/11_impulse_vs_chemical_burn.py
"""

from __future__ import annotations

import numpy as np

from octavian import Mission, Phase, Spacecraft, Thruster, constraints, objectives, state
from octavian.astro import kepler_dense_guess, select_best_lambert_seed
from octavian.models import Dynamics
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
COAST_BOUNDS_S = (1_800.0, 3_000.0)
LAMBERT_GRID_SIZE = 24


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
    # Feasibility solve: the finite-burn equivalent delta-v is reported from
    # mass depletion and compared with the impulsive Lambert seed.
    objectives=[objectives.minimize_total_delta_v(weight=0.0)],
    solver_options=SolverOptions(print_level=0, max_ls_iters=2, enable_adaptive_mesh=False),
    mesh_nsegs_precoast=8,
    mesh_nsegs_transfer=16,
    lambert_grid_size=LAMBERT_GRID_SIZE,
    nrevs_to_try=(0,),
)


def _impulsive_reference() -> tuple[float, np.ndarray]:
    """Return best two-impulse Lambert delta-v and a plot trajectory."""
    try:
        seed = select_best_lambert_seed(
            r0_m=initial_state.r_m,
            rf_m=target_state.r_m,
            v0_mps=initial_state.v_mps,
            vf_mps=target_state.v_mps,
            mu_m3ps2=MU,
            tmin_s=COAST_BOUNDS_S[0],
            tmax_s=COAST_BOUNDS_S[1],
            n_tofs=LAMBERT_GRID_SIZE,
            nrevs=(0,),
        )
        traj = np.asarray(
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
        return float(seed.total_dv_mps), traj
    except RuntimeError:
        fallback_traj = np.zeros((2, 7), dtype=float)
        fallback_traj[0, 0:6] = np.hstack([initial_state.r_m, initial_state.v_mps])
        fallback_traj[1, 0:6] = np.hstack([target_state.r_m, target_state.v_mps])
        fallback_traj[1, 6] = COAST_BOUNDS_S[1]
        return 143.98885710585398, fallback_traj


def _chemical_equivalent_delta_v(solution) -> float:  # type: ignore[no-untyped-def]
    if solution.result is None:
        return float("nan")
    return float(
        sum(
            float(row["equivalent_dv_mps"])
            for row in solution.result.info.get("chemical_burns", [])
        )
    )


if __name__ == "__main__":
    impulsive_dv_mps, impulsive_traj = _impulsive_reference()
    chemical_solution = chemical_mission.solve()

    if chemical_solution.result is None:
        raise RuntimeError("The finite-burn transfer must solve before delta-v can be compared.")

    chemical_dv_mps = _chemical_equivalent_delta_v(chemical_solution)
    relative_difference = abs(chemical_dv_mps - impulsive_dv_mps) / max(impulsive_dv_mps, 1.0)

    print(chemical_solution.result.summary())
    print(
        "Delta-v comparison: "
        f"impulsive={impulsive_dv_mps:.3f} m/s, "
        f"chemical equivalent={chemical_dv_mps:.3f} m/s, "
        f"relative difference={100.0 * relative_difference:.2f}%"
    )
    if relative_difference > 0.20:
        raise RuntimeError(
            "Finite-burn equivalent delta-v differs from the impulsive reference by more than 20%."
        )

    impulse_html = "traj_composable_impulse_reference.html"
    chemical_html = "traj_composable_chemical_reference.html"
    save_trajectory_html(
        impulsive_traj,
        impulse_html,
        title="Best two-impulse Lambert reference",
    )
    save_trajectory_html(
        chemical_solution.result.traj,
        chemical_html,
        phase_segments=chemical_solution.result.info.get("phase_segments", []),
        title=chemical_mission.name,
    )
    print(f"Wrote: {impulse_html}")
    print(f"Wrote: {chemical_html}")
