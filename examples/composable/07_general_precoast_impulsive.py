"""Example 05: True composable compiler — precoast + impulsive link + terminal impulse objective.

This example uses the *general* composable layout:

- Boundary constraints are explicit objects (constraints.state / constraints.position)
- Impulsive maneuvers are explicit variables (variables.ImpulsiveDeltaV)
- Linking is explicit (links.position() / links.state())

Key semantics:
- constraints.state(xf, where="Back") wants to fix both R and V
- BUT if you also declare ImpulsiveDeltaV(where="Back") on that phase, Octavian
  will *relax the V constraint* and instead add a terminal Δv objective:
      || V_back - V_desired ||
  while still fixing final position.

This avoids hard-coding a "two_impulse_precoast" solver shape — the mission is compiled
directly to ASSET.
"""

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
from octavian.viz import save_trajectory_html

MU = 3.986004418e14

spacecraft = Spacecraft(
    name="DemoSat",
    dry_mass_kg=150.0,
    thrusters=[Thruster(name="main", thrust_N=0.0, isp_s=1e9)],  # not used for impulsive
)

dynamics = Dynamics(mu_m3ps2=MU)

x0 = state(
    r_m=[7000e3, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
)

xf = state(
    r_m=[6100e3, 5000e3, 0.0],
    v_mps=[-1500.0, 4500.0, 0.0],
)

precoast = Phase(
    name="precoast",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    tof_bounds_s=(0.0, 6000.0),  # interpreted as absolute bound on t1 (seconds)
    constraints=[constraints.state(x0, where="Front")],
)

transfer = Phase(
    name="transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=precoast,
    tof_bounds_s=(400.0, 60000.0),  # interpreted as absolute bound on tf (seconds)
    link=links.impulsive(),
    constraints=[constraints.state(xf, where="Back")],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),  # link maneuver objective
        variables.ImpulsiveDeltaV(where="Back"),   # terminal maneuver objective
    ],
)

mission = Mission(
    phases=[precoast, transfer],
    name="Composable mission: precoast + impulsive link + terminal Δv",
    objectives=[objectives.minimize_total_delta_v()],
)

sol = mission.solve()
print(sol.summary())

if sol.result is not None:
    save_trajectory_html(sol.result.traj, "traj_composable_general.html", title=mission.name, maneuvers=sol.result.maneuvers)
    print("Wrote traj_composable_general.html")
