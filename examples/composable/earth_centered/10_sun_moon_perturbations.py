"""Earth-centered composable example 10: coast with J2, Moon, and Sun.

Run:
  python examples/composable/earth_centered/10_sun_moon_perturbations.py
"""

from __future__ import annotations

import numpy as np

from octavian import Mission, Phase, Spacecraft, Thruster, constraints, objectives, state, variables
from octavian.models import Dynamics, Perturbations
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14


spacecraft = Spacecraft(
    name="Perturbed coast demo spacecraft",
    dry_mass_kg=500.0,
    thrusters=[Thruster(name="main", thrust_N=0.0, isp_s=300.0, propellant_mass_kg=0.0)],
)

radius_m = 7_000e3
circular_speed_mps = float(np.sqrt(MU / radius_m))

initial_state = state(
    r_m=[radius_m, 0.0, 0.0],
    v_mps=[0.0, circular_speed_mps, 0.0],
)

target_position_m = np.array([-6_363_961.030678928, 2_899_711.648727294, 0.0])

dynamics = Dynamics(
    mu_m3ps2=MU,
    perturbations=Perturbations(j2=True, moon=True, sun=True),
    # One-hour samples are a practical default for short Earth-orbit examples.
    third_body_table_step_s=3600.0,
)

transfer = Phase(
    name="perturbed_transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    initial_state=initial_state,
    tof_bounds_s=(2_000.0, 4_000.0),
    constraints=[
        constraints.state(initial_state, where="Front"),
        constraints.position(target_position_m, where="Back"),
    ],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)

mission = Mission(
    name="Composable: J2, Moon, and Sun perturbed coast",
    initial_epoch="2026-01-01T00:00:00Z",
    phases=[transfer],
    objectives=[objectives.minimize_total_delta_v()],
    solver_options=SolverOptions(print_level=0, enable_adaptive_mesh=False),
    mesh_nsegs_transfer=40,
    lambert_grid_size=40,
    nrevs_to_try=(0,),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The perturbed transfer did not return a result.")

print(solution.result.summary())
output_path = "traj_composable_sun_moon_perturbations.html"
save_trajectory_html(
    solution.result.traj,
    output_path,
    maneuvers=solution.result.maneuvers,
    title=mission.name,
)
print(f"Wrote: {output_path}")
