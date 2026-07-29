"""Earth-centered composable example 04: hard versus objective terminal velocity.

This example uses a geometry that is feasible by pure Keplerian coast:
start and end are the same circular orbit, separated by a true-anomaly rotation.

We solve two missions:
  A) Hard terminal state: constraints.state(xf, Back), no ImpulsiveDeltaV(Back)
  B) Terminal dv objective: constraints.state(xf, Back) + ImpulsiveDeltaV(Back)
     (compiler relaxes V and adds an objective term)

Run:
  python examples/composable/earth_centered/04_terminal_velocity_hard_vs_objective.py
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
    objectives,
    variables,
)
from octavian.quick import state
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14


def rotz(theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


spacecraft = Spacecraft(name="DemoSat", dry_mass_kg=150.0, thrusters=[Thruster(name="main")])
dynamics = Dynamics(mu_m3ps2=MU)

r0 = np.array([7000e3, 0.0, 0.0])
v0 = np.array([0.0, float(np.sqrt(MU / 7000e3)), 0.0])
x0 = state(r_m=r0, v_mps=v0)

theta = np.deg2rad(35.0)
R = rotz(theta)
rf = R @ r0
vf = R @ v0
xf = state(r_m=rf, v_mps=vf)

# Expected mean motion n = sqrt(mu/a^3), coast time ~ theta/n
a = float(np.linalg.norm(r0))
n = float(np.sqrt(MU / a**3))
tof_guess = float(theta / n)

tof_bounds = (0.5 * tof_guess, 1.5 * tof_guess)


def solve_case(tag: str, terminal_is_objective: bool) -> None:
    vars_ = [variables.ImpulsiveDeltaV(where="Back")] if terminal_is_objective else []

    phase = Phase(
        name="coast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=tof_bounds,
        constraints=[
            constraints.state(x0, where="Front"),
            constraints.state(xf, where="Back"),
        ],
        variables=vars_,
    )

    mission = Mission(
        name=f"Composable terminal V: {tag}",
        phases=[phase],
        objectives=[objectives.minimize_total_delta_v()],
    )

    sol = mission.solve()
    print("\n" + "=" * 80)
    print(sol.summary())

    traj = sol.result.traj
    v_end = traj[-1, 3:6]
    verr = float(np.linalg.norm(v_end - vf))
    print(f"terminal velocity error vs desired: {verr:.6e} m/s")

    out_html = f"traj_composable_terminal_velocity_{tag}.html"
    save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
    print(f"Wrote: {out_html}")


solve_case("hard", terminal_is_objective=False)
solve_case("objective", terminal_is_objective=True)
