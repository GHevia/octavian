"""Relative example 26: free-time targeting in classical relative elements.

This is the singular classical-difference counterpart to example 20's
quasi-nonsingular D'Amico formulation. The optimizer propagates
``[Δa, Δe, Δi, ΔΩ, Δω, ΔM]`` directly and selects the time at which a target
relative mean anomaly is reached.

Run:
  python examples/composable/relative/26_classical_relative_elements.py
"""

from __future__ import annotations

from octavian import EARTH, Dynamics, Mission, Phase, Spacecraft, constraints, state
from octavian.astro import classical_to_cartesian
from octavian.relative import (
    ClassicalRelativeOrbitalElements,
    classical_relative_orbital_elements_to_relative_state,
    propagate_relative_orbital_elements,
    propagate_two_body_state,
)
from octavian.solvers import SolverOptions

CHIEF_SEMI_MAJOR_AXIS_M = 7_000_000.0
chief_position, chief_velocity = classical_to_cartesian(
    a_m=CHIEF_SEMI_MAJOR_AXIS_M,
    e=0.01,
    inc_deg=40.0,
    raan_deg=20.0,
    argp_deg=30.0,
    true_anomaly_deg=10.0,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief_eci = state(chief_position, chief_velocity)
initial_elements = ClassicalRelativeOrbitalElements(
    delta_a_m=500.0,
    delta_e=1.0e-4,
    delta_i_rad=2.0e-4,
    delta_raan_rad=-1.0e-4,
    delta_argp_rad=3.0e-4,
    delta_mean_anomaly_rad=-8.0e-4,
)
target_time_s = 1_800.0
target_elements = propagate_relative_orbital_elements(
    initial_elements,
    [target_time_s],
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    representation="classical_elements",
)[0, 0:6]

initial_ric = classical_relative_orbital_elements_to_relative_state(
    chief_eci,
    initial_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)
target_chief = propagate_two_body_state(
    chief_eci,
    target_time_s,
    EARTH.mu_m3ps2,
)
target_ric_guess = classical_relative_orbital_elements_to_relative_state(
    target_chief,
    target_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)

phase = Phase(
    name="classical_element_free_time",
    mode="relative_coast",
    spacecraft=Spacecraft(name="Deputy", dry_mass_kg=150.0),
    dynamics=Dynamics.relative(
        chief_initial_state_eci=chief_eci,
        propagation_mode="classical_elements",
    ),
    initial_state=initial_ric,
    final_state=target_ric_guess,
    tof_bounds_s=(1_500.0, 2_100.0),
    constraints=[
        constraints.relative_orbital_elements(
            initial_elements,
            where="Front",
        ),
        constraints.relative_orbital_element(
            "delta_mean_anomaly",
            float(target_elements[5]),
            representation="classical_elements",
            where="Back",
        ),
    ],
)

mission = Mission(
    name="Native classical relative-element free-time target",
    phases=[phase],
    objectives=[],
    mesh_nsegs_transfer=10,
    solver_options=SolverOptions(print_level=0),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The classical-element mission did not return a result.")

print(solution.summary())
print(f"Selected arrival time: {solution.result.tf_s():.6f} s")
print(f"Native layout: {solution.result.info['state_layouts'][0]}")
print(f"Constraint report: {solution.result.info['constraint_report']}")
solution.viz().save_html(
    "traj_classical_relative_elements.html",
    title=mission.name,
)
solution.viz().save_diagnostics_html(
    "diagnostics_classical_relative_elements.html",
    title=mission.name,
)
