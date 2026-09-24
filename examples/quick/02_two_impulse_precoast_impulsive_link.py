"""Quick example 02: precoast plus circular-orbit transfer.

Run:
  python examples/quick/02_two_impulse_precoast_impulsive_link.py

Outputs:
  - prints a short solution summary
  - writes a Plotly HTML trajectory with maneuver markers
"""

from __future__ import annotations

import numpy as np

from octavian import propagate, state, two_burn_rendezvous
from octavian.solvers import SolverOptions
from octavian.viz.matplotlib import save_trajectory_image
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
R_INITIAL_M = 7_000e3
R_FINAL_M = 12_000e3


x0 = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)

# Same circular target as example 01, with an optional loiter before transfer.
xf = state(
    r_m=[-R_FINAL_M, 0.0, 0.0],
    v_mps=[0.0, -float(np.sqrt(MU / R_FINAL_M)), 0.0],
)

mission = two_burn_rendezvous(
    x0,
    xf,
    mu_m3ps2=MU,
    precoast=True,
    t1_bounds_s=(1.0, 1_000.0),
    tf_bounds_s=(1_200.0, 12_000.0),
    nsegs=60,
    precoast_grid_size=12,
    lambert_grid_size=50,
    solver_options=SolverOptions(print_level=3),
    nrevs_to_try=(0,),
    name="Quick: precoast plus circular-orbit transfer",
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

out_html = "traj_quick_precoast_circular_transfer.html"
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
