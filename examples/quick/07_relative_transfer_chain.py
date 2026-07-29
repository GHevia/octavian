"""Quick example 07: string together relative transfers and coasts.

Two ordinary two-burn transfers are separated by a bounded coast on the first
target orbit. The optimizer therefore reports four impulses: depart/arrive at
the first target, then depart/arrive at the second target.
"""

from __future__ import annotations

import numpy as np

from octavian import EARTH, relative_transfer_chain, state
from octavian.solvers import SolverOptions

CHIEF_RADIUS_M = EARTH.mean_radius_m + 400_000.0
chief_eci = state(
    [CHIEF_RADIUS_M, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / CHIEF_RADIUS_M), 0.0],
)
initial_ric = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
inspection_point_ric = state([0.0, -500.0, 100.0], [0.0, 0.0, 0.0])
final_ric = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])

mission = relative_transfer_chain(
    initial_ric,
    [inspection_point_ric, final_ric],
    chief_initial_state_eci=chief_eci,
    transfer_time_bounds_s=[
        (600.0, 1_200.0),
        (600.0, 1_200.0),
    ],
    coast_time_bounds_s=(300.0, 600.0),
    nsegs_coast=16,
    nsegs_transfer=24,
    seed_grid_size=24,
    solver_options=SolverOptions(print_level=0),
    name="Quick: chained relative transfers",
)

solution = mission.solve()
print(solution.summary())
solution.viz().save_html(
    "traj_quick_relative_transfer_chain.html",
    title=mission.name,
)
solution.viz().save_diagnostics_html(
    "diagnostics_quick_relative_transfer_chain.html",
    title="Chained relative transfer RIC history",
)
