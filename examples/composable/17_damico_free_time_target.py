"""Composable example 17: target a native D'Amico ROE at free arrival time.

The Cartesian RIC states are guess anchors only. The optimizer propagates
D'Amico elements directly, fixes all six initial ROEs, and selects the arrival
time at which the requested ``delta_lambda`` is reached.
"""

from __future__ import annotations

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    constraints,
    state,
)
from octavian.astro import classical_to_cartesian
from octavian.relative import (
    RelativeOrbitalElements,
    propagate_relative_orbital_elements,
    propagate_two_body_state,
    relative_orbital_elements_to_relative_state,
)
from octavian.solvers import SolverOptions

chief_position, chief_velocity = classical_to_cartesian(
    a_m=7_000_000.0,
    e=0.001,
    inc_deg=40.0,
    raan_deg=20.0,
    argp_deg=10.0,
    true_anomaly_deg=30.0,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief_eci = state(chief_position, chief_velocity)
initial_elements = RelativeOrbitalElements(
    delta_a=2.0e-4,
    delta_lambda_rad=-0.004,
    delta_ex=1.0e-4,
    delta_ey=-2.0e-4,
    delta_ix_rad=1.0e-4,
    delta_iy_rad=2.0e-4,
)
target_time_s = 1_800.0
target_elements = propagate_relative_orbital_elements(
    initial_elements,
    [target_time_s],
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
)[0]

initial_ric = relative_orbital_elements_to_relative_state(
    chief_eci,
    initial_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)
target_chief_eci = propagate_two_body_state(
    chief_eci,
    target_time_s,
    EARTH.mu_m3ps2,
)
target_ric_guess = relative_orbital_elements_to_relative_state(
    target_chief_eci,
    target_elements[0:6],
    mu_m3ps2=EARTH.mu_m3ps2,
)

phase = Phase(
    name="damico_free_time",
    mode="relative_coast",
    spacecraft=Spacecraft(name="Deputy", dry_mass_kg=250.0),
    dynamics=Dynamics.relative(
        chief_initial_state_eci=chief_eci,
        propagation_mode="damico",
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
            "delta_lambda",
            float(target_elements[1]),
            where="Back",
        ),
    ],
)
mission = Mission(
    name="Composable: native D'Amico free-time target",
    phases=[phase],
    objectives=[],
    mesh_nsegs_transfer=10,
    lambert_grid_size=10,
    solver_options=SolverOptions(print_level=0),
)

solution = mission.solve()
print(solution.summary())
print(f"Selected arrival time: {solution.result.tf_s():.6f} s")
print(f"Native layout: {solution.result.info['state_layouts'][0]}")
print(f"Constraint report: {solution.result.info['constraint_report']}")

solution.viz().save_html(
    "traj_composable_damico_free_time.html",
    title="D'Amico relative-element free-time target",
)
solution.viz().save_diagnostics_html(
    "diagnostics_composable_damico_free_time.html",
    title="RIC view reconstructed from native D'Amico propagation",
)
