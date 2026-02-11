"""Composable example 05: plotting with maneuver markers.

This is intentionally focused on visualization.

Run:
  python examples/composable/05_plot_with_maneuvers.py
"""

from __future__ import annotations

import numpy as np

from octavian import Dynamics, Mission, Phase, Spacecraft, Thruster, constraints, links, objectives, variables
from octavian.quick import state
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14


def snap_maneuvers_to_traj(traj: np.ndarray, maneuvers):
    """Snap maneuver marker positions to the nearest trajectory sample by time.

    This prevents tiny visual offsets if the maneuver r_m isn't exactly on the returned polyline.
    """
    t = np.asarray(traj[:, 6], float)
    r = np.asarray(traj[:, 0:3], float)
    out = []
    for m in maneuvers:
        mt = float(m.t_s)
        i = int(np.argmin(np.abs(t - mt)))
        out.append(type(m)(r_m=r[i].copy(), t_s=mt, dv_mps=np.asarray(m.dv_mps, float).reshape(3), name=m.name))
    return out


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
    )

    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=precoast,
        link=links.impulsive(),
        tof_bounds_s=(400.0, 60_000.0),
        constraints=[constraints.state(xf, where="Back")],
        variables=[variables.ImpulsiveDeltaV(where="Front"), variables.ImpulsiveDeltaV(where="Back")],
    )

    mission = Mission(
        name="Composable: plotting maneuvers",
        phases=[precoast, transfer],
        objectives=[objectives.minimize_total_delta_v()],
    )

    sol = mission.solve()
    print(sol.summary())

    traj = sol.result.traj

    out1 = "traj_plot_maneuvers_raw.html"
    save_trajectory_html(traj, out1, maneuvers=sol.result.maneuvers, title=mission.name + " (raw maneuvers)")

    out2 = "traj_plot_maneuvers_snapped.html"
    snapped = snap_maneuvers_to_traj(traj, list(sol.result.maneuvers))
    save_trajectory_html(traj, out2, maneuvers=snapped, title=mission.name + " (snapped maneuvers)")

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")


if __name__ == "__main__":
    main()
