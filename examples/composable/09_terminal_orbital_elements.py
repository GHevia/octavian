"""Composable example 09: terminal orbital-element constraints.

This example fixes the departure state in Cartesian coordinates, but constrains the
arrival orbit by semi-major axis, eccentricity, and inclination instead of a single
terminal Cartesian state. The transfer still uses two impulsive burns: one at the
start and one at the end.

`final_state` is still supplied as a guess anchor for the Lambert-style seed search.
It is not a boundary position or velocity constraint here.
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
from octavian.astro import cartesian_to_classic, classical_to_cartesian
from octavian.quick import state
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14

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
    true_anomaly_deg=70.0,
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
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)

mission = Mission(
    name="Composable: terminal orbital-element constraints",
    phases=[transfer],
    objectives=[objectives.minimize_total_delta_v()],
)

sol = mission.solve()
print(sol.summary())

print("Applied terminal constraints:")
print(f"  semi_major_axis  target={target_a_m:.3f} m  tol=2000.000 m")
print(f"  eccentricity     target={target_e:.6f}     tol=0.005000")
print(f"  inclination_deg  target={target_inc_deg:.6f} tol=0.200000")
rf_m = sol.result.traj[-1, 0:3]
vf_mps = sol.result.traj[-1, 3:6]
if float(np.linalg.norm(rf_m)) > 0.0:
    final_oe = cartesian_to_classic(r_m=rf_m, v_mps=vf_mps, mu_m3ps2=MU)
    print("Achieved final orbital elements:")
    print(
        f"  a_m              value={final_oe['a_m']:.3f} m  error={final_oe['a_m'] - target_a_m:.3f} m"
    )
    print(f"  e                value={final_oe['e']:.6f}     error={final_oe['e'] - target_e:.6f}")
    print(
        f"  inc_deg          value={final_oe['inc_deg']:.6f} error={final_oe['inc_deg'] - target_inc_deg:.6f}"
    )

    final_oe2 = cartesian_to_classic(r_m=sol.result.traj[1, 0:3], v_mps=sol.result.traj[1, 3:6], mu_m3ps2=MU)
    print("Achieved final orbital elements just before:")
    print(
        f"  a_m              value={final_oe2['a_m']:.3f}   m"
    )
    print(f"  e                value={final_oe2['e']:.6f}   ")
    print(
        f"  inc_deg          value={final_oe2['inc_deg']:.6f} "
    )
    # print(sol.result.traj[-3:, 0:6])
else:
    print("Achieved final orbital elements: unavailable for zero-radius terminal state.")

out_html = "traj_composable_terminal_orbital_elements.html"
save_trajectory_html(sol.result.traj, out_html, maneuvers=sol.result.maneuvers, title=mission.name)
print(f"Wrote: {out_html}")
