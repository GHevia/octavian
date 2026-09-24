"""Earth-centered composable example 08: chemical burn-coast-burn with J2.

Run:
  python examples/composable/earth_centered/08_chemical_burn_j2.py
"""

from __future__ import annotations

import numpy as np

from octavian import Mission, Phase, Spacecraft, Thruster, constraints, objectives, propagate, state
from octavian.models import Dynamics, Perturbations
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14


spacecraft = Spacecraft(
    name="Demo spacecraft",
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

initial_state = state(
    r_m=[7000e3, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
)
target_state = state(
    # A nearby target on the same nominal orbit, with a modest terminal
    # velocity offset so the burn-coast-burn structure has real work to do.
    r_m=[-5_951_609.571397256, 3_684_880.3928557364, 0.0],
    v_mps=[-3_972.32911602311, -6_395.880426811104, 0.0],
)

dynamics = Dynamics(mu_m3ps2=MU, perturbations=Perturbations(j2=True))

departure_burn = Phase(
    name="departure_burn",
    mode="chemical_burn",
    spacecraft=spacecraft,
    dynamics=dynamics,
    initial_state=initial_state,
    tof_bounds_s=(20.0, 120.0),
    constraints=[
        constraints.state(initial_state, where="Front"),
    ],
)

coast = Phase(
    name="coast",
    mode="coast",
    spacecraft=spacecraft,
    previous=departure_burn,
    dynamics=dynamics,
    tof_bounds_s=(1_800.0, 3_000.0),
    tof_is_relative=True,
)

arrival_burn = Phase(
    name="arrival_burn",
    mode="chemical_burn",
    spacecraft=spacecraft,
    previous=coast,
    dynamics=dynamics,
    final_state=target_state,
    tof_bounds_s=(20.0, 120.0),
    tof_is_relative=True,
    constraints=[
        constraints.state(target_state, where="Back"),
    ],
)


mission = Mission(
    name="Composable: burn-coast-burn chemical transfer with J2",
    phases=[departure_burn, coast, arrival_burn],
    # Keep this first chemical-burn example as a feasibility solve. Propellant
    # usage is reported from the mass state after convergence.
    objectives=[objectives.minimize_propellant(weight=0.0)],
    solver_options=SolverOptions(print_level=0, max_ls_iters=2, enable_adaptive_mesh=False),
    mesh_nsegs_precoast=8,
    mesh_nsegs_transfer=16,
    lambert_grid_size=24,
    nrevs_to_try=(0,),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The finite-burn mission did not return a result.")

print(solution.result.summary())
for burn_summary in solution.result.info.get("chemical_burns", []):
    print(
        f"{burn_summary['phase']}: "
        f"propellant={burn_summary['propellant_used_kg']:.3f} kg, "
        f"equivalent dv={burn_summary['equivalent_dv_mps']:.3f} m/s"
    )

# Propagate the departure and target states for one nominal orbital period.
reference_trajectories = []
for name, orbit_state, color in (
    ("Departure orbit", initial_state, "#F59E0B"),
    ("Target orbit", target_state, "#C084FC"),
):
    semi_major_axis_m = 1.0 / (
        2.0 / np.linalg.norm(orbit_state.r_m) - np.dot(orbit_state.v_mps, orbit_state.v_mps) / MU
    )
    period_s = 2.0 * np.pi * np.sqrt(semi_major_axis_m**3 / MU)
    reference_trajectories.append(
        {
            "name": name,
            "traj": propagate.inertial(
                orbit_state, np.linspace(0.0, period_s, 361), perturbations=dynamics.perturbations
            ),
            "color": color,
        }
    )

output_path = "traj_composable_chemical_burn_j2.html"
save_trajectory_html(
    solution.result.traj,
    output_path,
    phase_segments=solution.result.info.get("phase_segments", []),
    reference_trajectories=reference_trajectories,
    title=mission.name,
)
print(f"Wrote: {output_path}")
