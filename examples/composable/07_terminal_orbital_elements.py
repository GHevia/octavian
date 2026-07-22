"""Composable example 07: terminal orbital-element constraints.

This example targets only semi-major axis, eccentricity, and inclination at the
end of the transfer. The terminal Cartesian state is provided only as a guess
anchor for seeding the solve. It reports one- and two-impulse local solutions;
the extra terminal burn expands a nonconvex optimization problem, so comparing
two independently optimized objective values is informative but not a global
optimality proof.

Run:
  python examples/composable/07_terminal_orbital_elements.py
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
from octavian.astro import classical_to_cartesian
from octavian.quick import state
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14


def _build_mission(*, use_terminal_burn: bool) -> Mission:
    spacecraft = Spacecraft(
        name="DemoSat",
        dry_mass_kg=150.0,
        thrusters=[Thruster(name="main")],
    )
    dynamics = Dynamics(mu_m3ps2=MU)

    initial_state = state(
        r_m=[7000e3, 0.0, 0.0],
        v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 250.0],
    )

    target_a_m = 8_400e3
    target_e = 0.18
    target_inc_deg = 28.5
    guess_r_m, guess_v_mps = classical_to_cartesian(
        a_m=target_a_m,
        e=target_e,
        inc_deg=target_inc_deg,
        raan_deg=35.0,
        argp_deg=20.0,
        true_anomaly_deg=70.0,
        mu_m3ps2=MU,
    )
    terminal_guess = state(r_m=guess_r_m, v_mps=guess_v_mps)

    phase_variables = [variables.ImpulsiveDeltaV(where="Front")]
    if use_terminal_burn:
        phase_variables.append(variables.ImpulsiveDeltaV(where="Back"))

    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        initial_state=initial_state,
        final_state=terminal_guess,
        tof_bounds_s=(1_200.0, 24_000.0),
        constraints=[
            constraints.state(initial_state, where="Front"),
            constraints.semi_major_axis(target_a_m, where="Back"),
            constraints.eccentricity(target_e, where="Back"),
            constraints.inclination_deg(target_inc_deg, where="Back"),
        ],
        variables=phase_variables,
    )

    burn_label = "two-impulse" if use_terminal_burn else "one-impulse"
    return Mission(
        name=f"Composable: terminal orbital elements ({burn_label})",
        phases=[transfer],
        objectives=[objectives.minimize_total_delta_v()],
    )


def _print_constraint_report(mission_label: str, solution) -> None:  # type: ignore[no-untyped-def]
    print(mission_label)
    print(solution.summary())
    report_rows = solution.result.info.get("constraint_report", []) if solution.result is not None else []
    if report_rows:
        print("  orbital-element constraints:")
        for row in report_rows:
            print(
                "    "
                f"{row['constraint']} @ {row['where']}: "
                f"target={row['target']:.6f} "
                f"actual={row['actual']:.6f} "
                f"error={row['error']:.6f} "
                f"ok={row['satisfied']}"
            )
    print()


one_impulse = _build_mission(use_terminal_burn=False)
two_impulse = _build_mission(use_terminal_burn=True)



initial_state = one_impulse.phases[0].initial_state
if initial_state is not None:
    print("Initial Cartesian state:")
    print(f"  r_m   = {np.asarray(initial_state.r_m, dtype=float)}")
    print(f"  v_mps = {np.asarray(initial_state.v_mps, dtype=float)}")
    print()

one_impulse_solution = one_impulse.solve()
two_impulse_solution = two_impulse.solve()

_print_constraint_report("One-impulse transfer", one_impulse_solution)
_print_constraint_report("Two-impulse transfer", two_impulse_solution)

if one_impulse_solution.result is not None and two_impulse_solution.result is not None:
    one_impulse_dv = one_impulse_solution.result.total_dv_mps()
    two_impulse_dv = two_impulse_solution.result.total_dv_mps()
    print(
        "Delta-v comparison: "
        f"one-impulse={one_impulse_dv:.6f} m/s, "
        f"two-impulse={two_impulse_dv:.6f} m/s"
    )
    print("  Note: independent local solves may converge to different minima.")

out_html = "traj_composable_terminal_orbital_elements.html"
save_trajectory_html(
    two_impulse_solution.result.traj,
    out_html,
    maneuvers=two_impulse_solution.result.maneuvers,
    title=two_impulse.name,
)
print(f"Wrote: {out_html}")
