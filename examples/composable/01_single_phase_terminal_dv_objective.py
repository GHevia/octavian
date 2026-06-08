"""Composable example 01: Hohmann transfer with terminal delta-v objective.

Key semantics:
  - constraints.state(xf, where="Back") would normally hard-fix (R,V)
  - variables.ImpulsiveDeltaV(where="Back") causes the composable compiler to:
      * hard-fix R at Back
      * relax V at Back
      * add objective term ||V_back - xf.v||

Run:
  python examples/composable/01_single_phase_terminal_dv_objective.py
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
    variables,
)
from octavian.quick import state
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
R_INITIAL_M = 7_000e3
R_FINAL_M = 12_000e3

spacecraft = Spacecraft(name="DemoSat", dry_mass_kg=150.0, thrusters=[Thruster(name="main")])
dynamics = Dynamics(mu_m3ps2=MU)

x0 = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)

# Opposite-side circular target: this is the Hohmann reference case used in tests.
xf = state(
    r_m=[-R_FINAL_M, 0.0, 0.0],
    v_mps=[0.0, -float(np.sqrt(MU / R_FINAL_M)), 0.0],
)

phase = Phase(
    name="transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    tof_bounds_s=(3_000.0, 7_000.0),
    constraints=[
        constraints.state(x0, where="Front"),
        constraints.state(xf, where="Back"),
        # constraints.min_radius(6000e3, where="Path"),
    ],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)

mission = Mission(
    name="Composable: Hohmann transfer with terminal dv objective",
    phases=[phase],
    objectives=[objectives.minimize_total_delta_v()],
    solver_options=SolverOptions(print_level=3),
    lambert_grid_size=60,
    nrevs_to_try=(0,),
)

sol = mission.solve()
print(sol.summary())

out_html = "traj_composable_hohmann_terminal_dv_objective.html"
save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
print(f"Wrote: {out_html}")
