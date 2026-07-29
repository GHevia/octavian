"""Earth-centered composable example 06: precoast plus a three-burn transfer.

Impulsive links mean (R,t) are continuous but V may jump.
This example has three impulses and a minimum-altitude path constraint:
  - transfer1 Front (link impulse)
  - transfer2 Front (link impulse)
  - transfer2 Back (terminal impulse)

Run:
  python examples/composable/earth_centered/06_precoast_impulsive_link_3burn.py
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
R_EARTH_M = 6378.1363e3
R_INITIAL_M = 7_000e3
R_FINAL_M = 12_000e3

spacecraft = Spacecraft(name="DemoSat", dry_mass_kg=150.0, thrusters=[Thruster(name="main")])
dynamics = Dynamics(mu_m3ps2=MU)

x0 = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)

xf = state(
    r_m=[-R_FINAL_M, 0.0, 0.0],
    v_mps=[0.0, -float(np.sqrt(MU / R_FINAL_M)), 0.0],
)

# Per-phase duration bounds (s). Set tof_is_relative=True so these are treated as durations.
precoast_dur = (0.0, 600.0)
transfer1_dur = (600.0, 30_000.0)
transfer2_dur = (600.0, 30_000.0)

min_altitude_m = 60e3
r_min_m = R_EARTH_M + min_altitude_m

precoast = Phase(
    name="precoast",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    tof_bounds_s=precoast_dur,
    tof_is_relative=True,
    constraints=[
        constraints.state(x0, where="Front"),
        constraints.min_radius(r_min_m, where="Path"),
    ],
)

transfer1 = Phase(
    name="transfer1",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=precoast,
    link=links.impulsive(),
    tof_bounds_s=transfer1_dur,
    tof_is_relative=True,
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
    ],
    constraints=[constraints.min_radius(r_min_m, where="Path")],
)

transfer2 = Phase(
    name="transfer2",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=transfer1,
    link=links.impulsive(),
    tof_bounds_s=transfer2_dur,
    tof_is_relative=True,
    constraints=[
        constraints.state(xf, where="Back"),
    ],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)

mission = Mission(
    name="Composable: precoast + two transfers (impulsive link)",
    phases=[precoast, transfer1, transfer2],
    objectives=[objectives.minimize_total_delta_v()],
)

sol = mission.solve()
print(sol.summary())

out_html = "traj_composable_precoast_impulsive_link_3burn.html"
save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
print(f"Wrote: {out_html}")
