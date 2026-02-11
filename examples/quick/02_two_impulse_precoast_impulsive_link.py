"""Quick example 02: precoast + two-impulse rendezvous (impulsive link).

Run:
  python examples/quick/02_two_impulse_precoast_impulsive_link.py

Outputs:
  - prints a short solution summary
  - writes a Plotly HTML trajectory with maneuver markers
"""

from __future__ import annotations

import numpy as np

from octavian import two_burn_rendezvous, state
from octavian.viz.plotly import save_trajectory_html
from octavian.solvers import SolverOptions

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
        precoast=True,
        t1_bounds_s=(0.0, 6000.0),
        tf_bounds_s=(400.0, 60000.0),
        nsegs=60,
        precoast_grid_size=12,
        lambert_grid_size=50,
        solver_options=SolverOptions(print_level=3),
        nrevs_to_try=(0, 1),
        name="Quick: precoast + transfer (impulsive link)",
    )

    sol = mission.solve()
    print(sol.summary())

    out_html = "traj_quick_precoast_impulsive.html"
    save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
