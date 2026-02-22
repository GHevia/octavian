"""Quick example 03: Δv vs time trade (same geometry, different objective weight).

Run:
  python examples/quick/03_time_tradeoff.py

Notes:
  The quick v0.x rendezvous solvers support an additional objective term
  w_time * tf. Increase w_time to bias toward shorter transfers.
"""

from __future__ import annotations

import numpy as np

from octavian import state, two_burn_rendezvous
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14


def build_x0_xf():
    x0 = state(
        r_m=[7000e3, 0.0, 0.0],
        v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
    )

    xf = state(
        r_m=[6500e3, 2200e3, 0.0],
        v_mps=[-900.0, 7200.0, 0.0],
    )
    return x0, xf


def main() -> None:
    x0, xf = build_x0_xf()

    missions = [
        ("dv_only", 0.0),
        ("dv_plus_time", 2.0),
    ]

    for tag, w_time in missions:
        mission = two_burn_rendezvous(
            x0,
            xf,
            mu_m3ps2=MU,
            tf_bounds_s=(600.0, 20_000.0),
            nsegs=60,
            lambert_grid_size=60,
            nrevs_to_try=(0, 1),
            w_time=w_time,
            solver_options=SolverOptions(print_level=3),
            name=f"Quick: two-impulse ({tag})",
        )

        sol = mission.solve()
        print("\n" + "=" * 80)
        print(sol.summary())

        out_html = f"traj_quick_time_tradeoff_{tag}.html"
        save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
        print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
