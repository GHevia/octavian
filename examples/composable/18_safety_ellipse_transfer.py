"""Composable example 18: transfer between two D'Amico ROE sets.

The departure and arrival conditions are authored as D'Amico relative orbital
elements (ROEs). Each set is converted to a Cartesian RIC boundary state at
its boundary epoch, then the transfer is optimized with exact coupled-ECI
dynamics and impulsive maneuvers at the two boundaries.

Native ``damico`` propagation is intentionally not used for the transfer:
that model describes unforced, two-body ROE drift and cannot represent the
instantaneous ROE changes produced by an impulse. ``coupled_eci`` instead
propagates the chief and deputy independently with J2 and solar gravity while
keeping the public states, constraints, maneuver components, and plots in RIC.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Perturbations,
    Phase,
    Spacecraft,
    constraints,
    objectives,
    state,
    variables,
)
from octavian.astro import classical_to_cartesian
from octavian.relative import (
    RelativeOrbitalElements,
    propagate_relative_elements_to_ric,
    propagate_relative_numerical,
    relative_orbital_elements_to_relative_state,
)
from octavian.solvers import SolverOptions
from octavian.viz import (
    save_relative_trajectory_html,
    save_trajectory_diagnostics_html,
)

PRE_COAST_DURATION_S = 6_000.0
POST_COAST_DURATION_S = 6_000.0
TRANSFER_DURATION_S = 1_800.0
# Phase duration is optimized, so use a narrow interval to prescribe the
# target epoch while still satisfying the API's strict lower < upper rule.
TRANSFER_TIME_TOLERANCE_S = 1.0e-3
CHIEF_SEMI_MAJOR_AXIS_M = 7_000_000.0
INITIAL_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
FORCE_MODEL = Perturbations(j2=True, sun=True)

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

# The two safety ellipses are specified directly as D'Amico ROEs.  The
# in-plane and out-of-plane components are dimensionless/radians; dividing
# desired meter-scale offsets by the chief semi-major axis makes that explicit.
departure_elements = RelativeOrbitalElements(
    delta_a=0.0,
    delta_lambda_rad=-7_000.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ex=700.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ey=-1_000.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ix_rad=700.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_iy_rad=-1_000.0 / CHIEF_SEMI_MAJOR_AXIS_M,
)
arrival_elements = RelativeOrbitalElements(
    delta_a=0.0,
    delta_lambda_rad=-1_000.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ex=200.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ey=100.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ix_rad=200.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_iy_rad=200.0 / CHIEF_SEMI_MAJOR_AXIS_M,
)

# An ROE set maps to a different RIC state as the chief advances. Propagate the
# chief to the target epoch with the same force model used by the optimizer.
departure_ric = relative_orbital_elements_to_relative_state(
    chief_eci,
    departure_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief_history = propagate_relative_numerical(
    chief_eci,
    None,
    [0.0, TRANSFER_DURATION_S],
    deputy_initial_eci=chief_eci,
    perturbations=FORCE_MODEL,
    initial_epoch=INITIAL_EPOCH,
    max_step_s=30.0,
    ephemeris_step_s=300.0,
)
arrival_chief_eci = state(
    chief_history.chief_states_eci[-1, 0:3],
    chief_history.chief_states_eci[-1, 3:6],
)
arrival_ric = relative_orbital_elements_to_relative_state(
    arrival_chief_eci,
    arrival_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)
transfer = Phase(
    name="safety_ellipse_transfer",
    mode="relative_coast",
    spacecraft=Spacecraft(name="Deputy", dry_mass_kg=250.0),
    dynamics=Dynamics.relative(
        chief_name="Chief",
        chief_initial_state_eci=chief_eci,
        propagation_mode="coupled_eci",
        perturbations=FORCE_MODEL,
        third_body_table_step_s=300.0,
    ),
    initial_state=departure_ric,
    final_state=arrival_ric,
    tof_bounds_s=(
        TRANSFER_DURATION_S - TRANSFER_TIME_TOLERANCE_S,
        TRANSFER_DURATION_S + TRANSFER_TIME_TOLERANCE_S,
    ),
    constraints=[
        constraints.state(departure_ric, where="Front"),
        constraints.state(arrival_ric, where="Back"),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)
mission = Mission(
    name="Composable: safety-ellipse ROE transfer",
    phases=[transfer],
    objectives=[objectives.minimize_total_delta_v()],
    initial_epoch=INITIAL_EPOCH,
    mesh_nsegs_transfer=40,
    lambert_grid_size=60,
    solver_options=SolverOptions(print_level=0),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The safety-ellipse transfer did not return a result.")

print(solution.summary())
print(f"Selected transfer duration: {solution.result.tf_s():.6f} s")
print(f"Internal state layout: {solution.result.info['state_layouts'][0]}")
print(f"Constraint report: {solution.result.info['constraint_report']}")

# These two coasts are analysis-only: they visualize the natural motion before
# the departure impulse and after the arrival impulse without adding phases or
# decision variables to the optimization problem.
transfer_end_s = float(solution.result.tf_s())
pre_coast_times_s = np.linspace(-PRE_COAST_DURATION_S, 0.0, 121)
post_coast_elapsed_s = np.linspace(0.0, POST_COAST_DURATION_S, 121)
pre_coast_traj = propagate_relative_elements_to_ric(
    departure_elements,
    pre_coast_times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=FORCE_MODEL,
    initial_epoch=INITIAL_EPOCH,
    max_step_s=30.0,
    ephemeris_step_s=300.0,
)
post_coast_traj = propagate_relative_elements_to_ric(
    arrival_elements,
    post_coast_elapsed_s,
    chief_initial_state_eci=arrival_chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=FORCE_MODEL,
    initial_epoch=INITIAL_EPOCH + timedelta(seconds=TRANSFER_DURATION_S),
    max_step_s=30.0,
    ephemeris_step_s=300.0,
)

# Shift the zero-based histories onto one mission-elapsed plot axis beginning
# at the start of the pre-transfer coast.
pre_coast_traj[:, 6] += PRE_COAST_DURATION_S
transfer_plot_traj = solution.traj.copy()
transfer_plot_traj[:, 6] += PRE_COAST_DURATION_S
post_coast_traj[:, 6] += PRE_COAST_DURATION_S + transfer_end_s

# Drop repeated phase-boundary samples before stitching the plot history.
plot_traj = np.vstack(
    [
        pre_coast_traj[:-1],
        transfer_plot_traj,
        post_coast_traj[1:],
    ]
)
phase_segments = [
    {
        "name": "Initial safety ellipse (propagated)",
        "t_start_s": 0.0,
        "t_end_s": PRE_COAST_DURATION_S,
        "color": "#636EFA",
    },
    {
        "name": "Two-impulse transfer (optimized)",
        "t_start_s": PRE_COAST_DURATION_S,
        "t_end_s": PRE_COAST_DURATION_S + transfer_end_s,
        "color": "#EF553B",
    },
    {
        "name": "Target safety ellipse (propagated)",
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
    "traj_safety_ellipse_transfer.html",
    phase_segments=phase_segments,
    title="Transfer between D'Amico safety ellipses",
)
save_trajectory_diagnostics_html(
    plot_traj,
    "diagnostics_safety_ellipse_transfer.html",
    frame_kind="relative",
    title="Safety-ellipse transfer state history",
)
