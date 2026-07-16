"""Quick example 01: Hohmann transfer between circular orbits.

Run:
  python examples/quick/01_two_impulse_free_time.py

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


# Define the departure and arrival states in SI units. This opposite-side
# geometry has the familiar analytical Hohmann transfer as its optimum.
initial_state = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)
target_state = state(
    r_m=[-R_FINAL_M, 0.0, 0.0],
    v_mps=[0.0, -float(np.sqrt(MU / R_FINAL_M)), 0.0],
)

# The quick API turns this configuration into a complete two-impulse mission.
mission = two_burn_rendezvous(
    initial_state,
    target_state,
    mu_m3ps2=MU,
    tf_bounds_s=(3_000.0, 7_000.0),
    nsegs=60,
    lambert_grid_size=60,
    nrevs_to_try=(0,),
    solver_options=SolverOptions(print_level=3),
    name="Quick: Hohmann transfer between circular orbits",
)

# Solve, inspect, and plot. Octavian examples intentionally read like flat
# mission configuration files, so each line can be edited in place.
solution = mission.solve()
print(solution.summary())

output_path = "traj_quick_hohmann_transfer.html"
save_trajectory_html(
    solution.result.traj,
    output_path,
    maneuvers=solution.result.maneuvers,
    title=mission.name,
)
print(f"Wrote: {output_path}")
