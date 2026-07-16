"""Quick example 01: Hohmann transfer between circular orbits.

Run:
  python examples/quick/01_two_impulse_free_time.py

Outputs:
  - prints a short solution summary
  - writes a Plotly HTML trajectory with maneuver markers
"""

from __future__ import annotations

import numpy as np

from octavian import Mission, state, two_burn_rendezvous
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
R_INITIAL_M = 7_000e3
R_FINAL_M = 12_000e3


def build_mission() -> Mission:
    """Build the canonical quick two-impulse Hohmann transfer."""
    initial_state = state(
        r_m=[R_INITIAL_M, 0.0, 0.0],
        v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
    )

    # Opposite-side circular target. The analytical Hohmann solution is the
    # reference used by the regression tests for this example.
    final_state = state(
        r_m=[-R_FINAL_M, 0.0, 0.0],
        v_mps=[0.0, -float(np.sqrt(MU / R_FINAL_M)), 0.0],
    )

    return two_burn_rendezvous(
        initial_state,
        final_state,
        mu_m3ps2=MU,
        tf_bounds_s=(3_000.0, 7_000.0),
        nsegs=60,
        lambert_grid_size=60,
        nrevs_to_try=(0,),
        solver_options=SolverOptions(print_level=3),
        name="Quick: Hohmann transfer between circular orbits",
    )


mission = build_mission()


def main() -> None:
    """Solve the example and write its interactive trajectory plot."""
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


if __name__ == "__main__":
    main()
