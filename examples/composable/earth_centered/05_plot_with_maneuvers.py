"""Earth-centered composable example 05: plotting with maneuver markers.

This is intentionally focused on visualization.

Run:
  python examples/composable/earth_centered/05_plot_with_maneuvers.py
"""

from __future__ import annotations

from dataclasses import replace

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
R_FINAL_M = 12_000e3


spacecraft = Spacecraft(name="DemoSat", dry_mass_kg=150.0, thrusters=[Thruster(name="main")])
dynamics = Dynamics(mu_m3ps2=MU)

R_EARTH_M = 6378.1363e3
min_altitude_m = 60e3
r_min_m = R_EARTH_M + min_altitude_m

x0 = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)

xf = state(
    r_m=[-R_FINAL_M, 0.0, 0.0],
    v_mps=[0.0, -float(np.sqrt(MU / R_FINAL_M)), 0.0],
)

precoast = Phase(
    name="precoast",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    tof_bounds_s=(1.0, 600.0),
    constraints=[constraints.state(x0, where="Front")],
)

transfer = Phase(
    name="transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=precoast,
    link=links.impulsive(),
    tof_bounds_s=(600.0, 60_000.0),
    constraints=[
        constraints.state(xf, where="Back"),
        constraints.min_radius(r_min_m, where="Path"),
    ],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)

mission = Mission(
    name="Composable: plotting maneuvers",
    phases=[precoast, transfer],
    objectives=[objectives.minimize_total_delta_v()],
)

sol = mission.solve()
print(sol.summary())

traj = sol.result.traj

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

out1 = "traj_plot_maneuvers_raw.html"
save_trajectory_html(
    traj,
    out1,
    maneuvers=sol.result.maneuvers,
    reference_trajectories=reference_trajectories,
    title=mission.name + " (raw maneuvers)",
)

out2 = "traj_plot_maneuvers_snapped.html"
# Align each marker with the nearest sampled trajectory position.
snapped = []
for maneuver in sol.result.maneuvers:
    index = int(np.argmin(np.abs(traj[:, 6] - maneuver.t_s)))
    snapped.append(replace(maneuver, r_m=traj[index, :3].copy()))
save_trajectory_html(
    traj,
    out2,
    maneuvers=snapped,
    reference_trajectories=reference_trajectories,
    title=mission.name + " (snapped maneuvers)",
)

print(f"Wrote: {out1}")
print(f"Wrote: {out2}")

save_trajectory_image(
    sol.traj,
    out1.replace(".html", ".png"),
    maneuvers=sol.result.maneuvers,
    reference_trajectories=reference_trajectories,
    projection="xy",
    title="Departure and target reference orbits — XY plane",
)
