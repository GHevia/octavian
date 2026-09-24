"""Earth-centered composable example 02: precoast with a continuous link.

Continuous link means (R,V,t) are continuous across the phase boundary.
So there is no link delta-v maneuver.

Run:
  python examples/composable/earth_centered/02_precoast_continuous_link.py
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
    propagate,
    variables,
)
from octavian.quick import state
from octavian.viz.matplotlib import save_trajectory_image
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
    variables=[variables.ImpulsiveDeltaV(where="Front")],
)

transfer = Phase(
    name="transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=precoast,
    link=links.continuous(),
    tof_bounds_s=(400.0, 60_000.0),
    constraints=[constraints.state(xf, where="Back")],
    variables=[variables.ImpulsiveDeltaV(where="Back")],
)

mission = Mission(
    name="Composable: precoast + transfer (continuous link)",
    phases=[precoast, transfer],
    objectives=[objectives.minimize_total_delta_v()],
)

sol = mission.solve()
print(sol.summary())

# Propagate the departure and target states for one nominal orbital period.
reference_trajectories = []
for name, orbit_state, color in (
    ("Departure orbit", x0, "#F59E0B"),
    ("Target orbit", xf, "#C084FC"),
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

out_html = "traj_composable_precoast_continuous_link.html"
save_trajectory_html(
    sol.result.traj,
    out_html,
    maneuvers=sol.result.maneuvers,
    reference_trajectories=reference_trajectories,
    title=mission.name,
)
print(f"Wrote: {out_html}")

save_trajectory_image(
    sol.traj,
    out_html.replace(".html", ".png"),
    maneuvers=sol.result.maneuvers,
    reference_trajectories=reference_trajectories,
    projection="xy",
    title="Departure and target reference orbits — XY plane",
)
