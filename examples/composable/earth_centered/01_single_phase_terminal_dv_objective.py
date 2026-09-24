"""Earth-centered composable example 01: Hohmann transfer with terminal delta-v objective.

Key semantics:
  - constraints.state(xf, where="Back") would normally hard-fix (R,V)
  - variables.ImpulsiveDeltaV(where="Back") causes the composable compiler to:
      * hard-fix R at Back
      * relax V at Back
      * add objective term ||V_back - xf.v||

Run:
  python examples/composable/earth_centered/01_single_phase_terminal_dv_objective.py
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
    objectives,
    propagate,
    variables,
)
from octavian.quick import state
from octavian.solvers import SolverOptions
from octavian.viz.matplotlib import save_trajectory_image
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
R_INITIAL_M = 7_000e3
R_FINAL_M = 12_000e3

# The composable API spells out the same mission in four layers: vehicle,
# dynamics, boundary states, and phase intent.
spacecraft = Spacecraft(
    name="DemoSat",
    dry_mass_kg=150.0,
    thrusters=[Thruster(name="main")],
)
dynamics = Dynamics(mu_m3ps2=MU)

initial_state = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)
target_state = state(
    r_m=[-R_FINAL_M, 0.0, 0.0],
    v_mps=[0.0, -float(np.sqrt(MU / R_FINAL_M)), 0.0],
)

transfer = Phase(
    name="transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    tof_bounds_s=(3_000.0, 7_000.0),
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
    name="Composable: Hohmann transfer with terminal dv objective",
    phases=[transfer],
    objectives=[objectives.minimize_total_delta_v()],
    solver_options=SolverOptions(print_level=3),
    lambert_grid_size=60,
    nrevs_to_try=(0,),
)

solution = mission.solve()
print(solution.summary())

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
            "traj": propagate.inertial(orbit_state, np.linspace(0.0, period_s, 361)),
            "color": color,
        }
    )

output_path = "traj_composable_hohmann_terminal_dv_objective.html"
save_trajectory_html(
    solution.result.traj,
    output_path,
    reference_trajectories=reference_trajectories,
    maneuvers=solution.result.maneuvers,
    title=mission.name,
)
print(f"Wrote: {output_path}")

image_path = "traj_composable_hohmann_terminal_dv_objective.png"
save_trajectory_image(
    solution.traj,
    image_path,
    reference_trajectories=reference_trajectories,
    maneuvers=solution.result.maneuvers,
    phase_segments=solution.result.info.get("phase_segments"),
    projection="xy",
    title=f"{mission.name} — XY plane",
)
print(f"Wrote: {image_path}")
