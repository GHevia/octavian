"""Quick example 02: precoast plus circular-orbit transfer.

Run:
  python examples/quick/02_two_impulse_precoast_impulsive_link.py

Outputs:
  - prints a short solution summary
  - writes a Plotly HTML trajectory with maneuver markers
"""

from __future__ import annotations

import numpy as np

from octavian import state, two_burn_rendezvous
from octavian.solvers import SolverOptions
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

out_html = "traj_quick_precoast_circular_transfer.html"
save_trajectory_html(
    sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name
)
print(f"Wrote: {out_html}")
