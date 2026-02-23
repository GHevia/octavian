"""Quick example 01: two-impulse rendezvous (single-phase coast).

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


def main() -> None:
    x0 = state(
        r_m=[7000e3, 0.0, 0.0],
        v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
    )

    # A reachable target a little ahead in-track (toy example)
    xf = state(
        r_m=[6900e3, 900e3, 0.0],
        v_mps=[0.0, 7500.0, 0.0],
    )

    mission = two_burn_rendezvous(
        x0,
        xf,
        mu_m3ps2=MU,
        tf_bounds_s=(600.0, 7200.0),
        nsegs=60,
        lambert_grid_size=60,
        nrevs_to_try=(0, 1),
        solver_options=SolverOptions(print_level=3),
        name="Quick: two-impulse (free time)",
    )

    sol = mission.solve()
    print(sol.summary())

    out_html = "traj_quick_two_impulse_free_time.html"
    save_trajectory_html(
        sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name
    )
    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
