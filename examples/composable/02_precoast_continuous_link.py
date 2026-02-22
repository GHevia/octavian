"""Composable example 02: precoast + transfer with a continuous link.

Continuous link means (R,V,t) are continuous across the phase boundary.
So there is no link Δv maneuver.

Run:
  python examples/composable/02_precoast_continuous_link.py
"""

from __future__ import annotations

import numpy as np

from octavian import (
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    Thruster,
    constraints,
    links,
    objectives,
    variables,
)
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

    xf = state(
        r_m=[6100e3, 5000e3, 0.0],
        v_mps=[-1500.0, 4500.0, 0.0],
    )

    precoast = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(0.0, 6000.0),
        constraints=[constraints.state(x0, where="Front")],
        variables=[variables.ImpulsiveDeltaV(where="Front")]
    )

    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=precoast,
        link=links.continuous(),
        tof_bounds_s=(400.0, 60_000.0),
        constraints=[constraints.state(xf, where="Back")],
        variables=[variables.ImpulsiveDeltaV(where="Back")],
    )

    mission = Mission(
        name="Composable: precoast + transfer (continuous link)",
        phases=[precoast, transfer],
        objectives=[objectives.minimize_total_delta_v()],
    )

    sol = mission.solve()
    print(sol.summary())

    out_html = "traj_composable_precoast_continuous_link.html"
    save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
