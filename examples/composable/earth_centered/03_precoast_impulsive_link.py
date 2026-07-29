"""Earth-centered composable example 03: precoast with an impulsive link.

Impulsive link means (R,t) are continuous but V may jump.
Declaring ImpulsiveDeltaV(where="Front") on the transfer phase
adds a link delta-v objective and produces a link maneuver marker.

Run:
  python examples/composable/earth_centered/03_precoast_impulsive_link.py
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
    links,
    objectives,
    variables,
)
from octavian.quick import state
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
R_INITIAL_M = 7_000e3
R_FINAL_M = 10_000e3
TARGET_TRUE_ANOMALY_RAD = np.deg2rad(140.0)

spacecraft = Spacecraft(name="DemoSat", dry_mass_kg=150.0, thrusters=[Thruster(name="main")])
dynamics = Dynamics(mu_m3ps2=MU)

x0 = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)

target_speed_mps = float(np.sqrt(MU / R_FINAL_M))
xf = state(
    r_m=[
        R_FINAL_M * float(np.cos(TARGET_TRUE_ANOMALY_RAD)),
        R_FINAL_M * float(np.sin(TARGET_TRUE_ANOMALY_RAD)),
        0.0,
    ],
    v_mps=[
        -target_speed_mps * float(np.sin(TARGET_TRUE_ANOMALY_RAD)),
        target_speed_mps * float(np.cos(TARGET_TRUE_ANOMALY_RAD)),
        0.0,
    ],
)

precoast = Phase(
    name="precoast",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    tof_bounds_s=(0.0, 6000.0),
    constraints=[constraints.state(x0, where="Front")],
)

transfer = Phase(
    name="transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=precoast,
    link=links.impulsive(),
    tof_bounds_s=(400.0, 60_000.0),
    constraints=[constraints.state(xf, where="Back")],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)

mission = Mission(
    name="Composable: precoast + transfer (impulsive link)",
    phases=[precoast, transfer],
    objectives=[objectives.minimize_total_delta_v()],
)

sol = mission.solve()
print(sol.summary())

out_html = "traj_composable_precoast_impulsive_link.html"
save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
print(f"Wrote: {out_html}")
