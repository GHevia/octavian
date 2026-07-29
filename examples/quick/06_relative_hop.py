"""Quick example 06: relative coast–burn–transfer–burn hop.

The helper builds two linked relative coast phases.  The first lets the deputy
choose a departure time; an impulsive link starts the transfer, and a terminal
impulse matches the requested target RIC velocity.
"""

from __future__ import annotations

import numpy as np

from octavian import EARTH, Perturbations, relative_hop, state
from octavian.solvers import SolverOptions

CHIEF_RADIUS_M = EARTH.mean_radius_m + 400_000.0
chief_eci = state(
    [CHIEF_RADIUS_M, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / CHIEF_RADIUS_M), 0.0],
)
initial_ric = state(
    [0.0, -1_000.0, 100.0],
    [0.0, 0.0, 0.0],
)
target_ric = state(
    [0.0, -100.0, 0.0],
    [0.0, 0.0, 0.0],
)

mission = relative_hop(
    initial_ric,
    target_ric,
    chief_initial_state_eci=chief_eci,
    departure_coast_time_bounds_s=(120.0, 600.0),
    transfer_time_bounds_s=(900.0, 1_800.0),
    # Quick relative missions use exact coupled dynamics by default. Force
    # models can be enabled without switching to the composable API.
    perturbations=Perturbations(j2=True),
    nsegs_coast=20,
    nsegs_transfer=30,
    seed_grid_size=30,
    solver_options=SolverOptions(print_level=0),
    name="Quick: perturbed relative hop",
)

solution = mission.solve()
print(solution.summary())
solution.viz().save_html(
    "traj_quick_relative_hop.html",
    title=mission.name,
)
solution.viz().save_diagnostics_html(
    "diagnostics_quick_relative_hop.html",
    title="Relative hop RIC history",
)
