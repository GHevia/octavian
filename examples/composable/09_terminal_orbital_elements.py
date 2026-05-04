"""Composable example 09: terminal orbital-element constraints.

This example fixes the departure state in Cartesian coordinates, but constrains the
arrival orbit by semi-major axis, eccentricity, and inclination instead of a single
terminal Cartesian state.

`final_state` is still supplied as a guess anchor for the Lambert-style seed search.
It is not a boundary constraint here.
"""

from __future__ import annotations

import numpy as np

from octavian import Dynamics, Mission, Phase, Spacecraft, Thruster, constraints, objectives, variables
from octavian.quick import state
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14


def classical_to_cartesian(
    *,
    a_m: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu_m3ps2: float,
):
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    nu = np.deg2rad(true_anomaly_deg)

    p = a_m * (1.0 - e**2)
    r_pf = (p / (1.0 + e * np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0.0], dtype=float)
    v_pf = np.sqrt(mu_m3ps2 / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0], dtype=float)

    cO = np.cos(raan)
    sO = np.sin(raan)
    ci = np.cos(inc)
    si = np.sin(inc)
    cw = np.cos(argp)
    sw = np.sin(argp)
    rot = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ],
        dtype=float,
    )
    return rot @ r_pf, rot @ v_pf


spacecraft = Spacecraft(name="DemoSat", dry_mass_kg=150.0, thrusters=[Thruster(name="main")])
dynamics = Dynamics(mu_m3ps2=MU)

x0 = state(
    r_m=[7000e3, 0.0, 0.0],
    v_mps=[0.0, 7200.0, 900.0],
)

target_a_m = 8_400e3
target_e = 0.18
target_inc_deg = 28.5
r_guess_m, v_guess_mps = classical_to_cartesian(
    a_m=target_a_m,
    e=target_e,
    inc_deg=target_inc_deg,
    raan_deg=35.0,
    argp_deg=20.0,
    true_anomaly_deg=50.0,
    mu_m3ps2=MU,
)
xf_guess = state(r_m=r_guess_m, v_mps=v_guess_mps)

transfer = Phase(
    name="transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    initial_state=x0,
    final_state=xf_guess,
    tof_bounds_s=(1_200.0, 24_000.0),
    constraints=[
        constraints.state(x0, where="Front"),
        constraints.semi_major_axis(target_a_m, where="Back", tol_m=2.0e3),
        constraints.eccentricity(target_e, where="Back", tol=5.0e-3),
        constraints.inclination_deg(target_inc_deg, where="Back", tol_deg=0.2),
    ],
    variables=[variables.ImpulsiveDeltaV(where="Front")],
)

mission = Mission(
    name="Composable: terminal orbital-element constraints",
    phases=[transfer],
    objectives=[objectives.minimize_total_delta_v()],
)

sol = mission.solve()
print(sol.summary())

out_html = "traj_composable_terminal_orbital_elements.html"
save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
print(f"Wrote: {out_html}")
