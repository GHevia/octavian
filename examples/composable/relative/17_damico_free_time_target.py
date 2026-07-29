"""Relative composable example 17: target a D'Amico ROE at free arrival time.

The Cartesian RIC states are guess anchors only. The optimizer propagates
D'Amico elements directly, fixes all six initial ROEs, and selects the arrival
time at which the requested ``delta_lambda`` is reached. Analysis-only
two-body coasts extend the optimized transfer before and after its boundaries
without adding either coast to the optimization problem.
"""

from __future__ import annotations

import numpy as np

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
    propagate_relative_elements_to_ric,
    propagate_relative_orbital_elements,
    propagate_two_body_state,
    relative_orbital_elements_to_relative_state,
)
from octavian.solvers import SolverOptions
from octavian.viz import (
    save_relative_trajectory_html,
    save_trajectory_diagnostics_html,
)

PRE_COAST_DURATION_S = 600.0
POST_COAST_DURATION_S = 600.0
CHIEF_SEMI_MAJOR_AXIS_M = 7_000_000.0

chief_position, chief_velocity = classical_to_cartesian(
    a_m=CHIEF_SEMI_MAJOR_AXIS_M,
    e=0.001,
    inc_deg=40.0,
    raan_deg=20.0,
    argp_deg=10.0,
    true_anomaly_deg=30.0,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief_eci = state(chief_position, chief_velocity)
initial_elements = RelativeOrbitalElements(
    delta_a=1_000 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_lambda_rad=-7000 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ex=700 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ey=-1000 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ix_rad=700 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_iy_rad=-1000 / CHIEF_SEMI_MAJOR_AXIS_M,
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
if solution.result is None:
    raise RuntimeError("The D'Amico free-time mission did not return a result.")

print(solution.summary())
print(f"Selected arrival time: {solution.result.tf_s():.6f} s")
print(f"Native layout: {solution.result.info['state_layouts'][0]}")
print(f"Constraint report: {solution.result.info['constraint_report']}")

# The optimizer only sees ``phase`` above. These coast histories are propagated
# analytically from the same D'Amico state and two-body chief reference.
transfer_end_s = float(solution.result.tf_s())
pre_coast_times_s = np.linspace(-PRE_COAST_DURATION_S, 0.0, 61)
post_coast_times_s = np.linspace(
    transfer_end_s,
    transfer_end_s + POST_COAST_DURATION_S,
    61,
)
pre_coast_traj = propagate_relative_elements_to_ric(
    initial_elements,
    pre_coast_times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
)
post_coast_traj = propagate_relative_elements_to_ric(
    initial_elements,
    post_coast_times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
)

# The transfer solve uses its own zero-based phase time. Shift all three
# histories onto one mission-elapsed plot axis that begins at the pre-coast.
pre_coast_traj[:, 6] += PRE_COAST_DURATION_S
transfer_plot_traj = solution.traj.copy()
transfer_plot_traj[:, 6] += PRE_COAST_DURATION_S
post_coast_traj[:, 6] += PRE_COAST_DURATION_S

# Drop the repeated phase-boundary samples before stitching the plot history.
plot_traj = np.vstack(
    [
        pre_coast_traj[:-1],
        transfer_plot_traj,
        post_coast_traj[1:],
    ]
)
phase_segments = [
    {
        "name": "Pre-transfer coast (propagated)",
        "t_start_s": 0.0,
        "t_end_s": PRE_COAST_DURATION_S,
        "color": "#636EFA",
    },
    {
        "name": "Free-time D'Amico transfer (optimized)",
        "t_start_s": PRE_COAST_DURATION_S,
        "t_end_s": PRE_COAST_DURATION_S + transfer_end_s,
        "color": "#EF553B",
    },
    {
        "name": "Post-transfer coast (propagated)",
        "t_start_s": PRE_COAST_DURATION_S + transfer_end_s,
        "t_end_s": (
            PRE_COAST_DURATION_S
            + transfer_end_s
            + POST_COAST_DURATION_S
        ),
        "color": "#00CC96",
    },
]

save_relative_trajectory_html(
    plot_traj,
    "traj_composable_damico_free_time.html",
    phase_segments=phase_segments,
    title="D'Amico relative-element free-time target",
)
save_trajectory_diagnostics_html(
    plot_traj,
    "diagnostics_composable_damico_free_time.html",
    frame_kind="relative",
    title="RIC view reconstructed from native D'Amico propagation",
)
