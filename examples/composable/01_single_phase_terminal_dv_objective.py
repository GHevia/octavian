"""Composable example 01: single coast phase with terminal Δv objective.

Key semantics:
  - constraints.state(xf, where="Back") would normally hard-fix (R,V)
  - variables.ImpulsiveDeltaV(where="Back") causes the composable compiler to:
      * hard-fix R at Back
      * relax V at Back
      * add objective term ||V_back - xf.v||

Run:
  python examples/composable/01_single_phase_terminal_dv_objective.py
"""

from __future__ import annotations

import numpy as np

from octavian import Dynamics, Mission, Phase, Spacecraft, Thruster, constraints, objectives, variables
from octavian.quick import state
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14


def main() -> None:
    spacecraft = Spacecraft(name="DemoSat", dry_mass_kg=150.0, thrusters=[Thruster(name="main")])
    dynamics = Dynamics(mu_m3ps2=MU)

    x0 = state(
        r_m=[7000e3, 0.0, 0.0],
        v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
    )

    # A "target" state: we'll fix rf, but only penalize vf via terminal Δv objective.
    xf = state(
        r_m=[6500e3, 2400e3, 0.0],
        v_mps=[-900.0, 7200.0, 0.0],
    )

    phase = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(60.0, 20_000.0),
        constraints=[
            constraints.state(x0, where="Front"),
            constraints.state(xf, where="Back"),
            constraints.min_radius(6000e3, where="Path"),
        ],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )

    mission = Mission(
        name="Composable: single-phase terminal dv objective",
        phases=[phase],
        objectives=[objectives.minimize_total_delta_v()],
    )

    sol = mission.solve()
    print(sol.summary())

    out_html = "traj_composable_single_phase_terminal_dv_objective.html"
    save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
