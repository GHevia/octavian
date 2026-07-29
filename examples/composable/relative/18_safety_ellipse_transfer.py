"""Relative composable example 18: free-time transfer between ROE-authored states.

The departure and arrival conditions are authored as D'Amico relative orbital
elements (ROEs). Each set is converted to a Cartesian RIC boundary state at
its nominal boundary epoch. An initial coast and the transfer duration are
then optimized independently with exact coupled-ECI dynamics.

Native ``damico`` propagation is intentionally not used for the transfer:
that model describes unforced, two-body ROE drift and cannot represent the
instantaneous ROE changes produced by an impulse. ``coupled_eci`` instead
propagates the chief and deputy independently with J2 and solar gravity while
keeping the public states, constraints, maneuver components, and plots in RIC.

The analysis-only coasts are anchored to the solved chief/deputy histories,
not the nominal ROE conversion states. The post-coast begins after applying
the solved terminal impulse.
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
    links,
    objectives,
    state,
    variables,
)
from octavian.astro import classical_to_cartesian
from octavian.relative import (
    RelativeOrbitalElements,
    propagate_relative_numerical,
    relative_orbital_elements_to_absolute_state,
    relative_orbital_elements_to_relative_state,
)
from octavian.solvers import SolverOptions
from octavian.types import Maneuver
from octavian.viz import (
    save_relative_trajectory_html,
    save_trajectory_diagnostics_html,
)

PRE_COAST_DURATION_S = 5_000.0
POST_COAST_DURATION_S = 5_000.0
INITIAL_COAST_BOUNDS_S = (120.0, 600.0)
TRANSFER_TIME_BOUNDS_S = (1_200.0, 2_400.0)
TARGET_MISSION_TIME_GUESS_S = 2_100.0
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

# An ROE set maps to a different RIC state as the chief advances. The nominal
# arrival epoch supplies a Cartesian seed/target while both phase durations
# remain free inside their declared bounds.
departure_ric = relative_orbital_elements_to_relative_state(
    chief_eci,
    departure_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)
departure_deputy_eci = relative_orbital_elements_to_absolute_state(
    chief_eci,
    departure_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief_history = propagate_relative_numerical(
    chief_eci,
    None,
    [0.0, TARGET_MISSION_TIME_GUESS_S],
    deputy_initial_eci=departure_deputy_eci,
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
deputy = Spacecraft(name="Deputy", dry_mass_kg=250.0)
dynamics = Dynamics.relative(
    chief_name="Chief",
    chief_initial_state_eci=chief_eci,
    propagation_mode="coupled_eci",
    perturbations=FORCE_MODEL,
    third_body_table_step_s=300.0,
    # Tables cover the cumulative absolute upper time plus this margin.
    third_body_table_margin_s=600.0,
)

initial_coast = Phase(
    name="initial_coast",
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    initial_state=departure_ric,
    tof_bounds_s=INITIAL_COAST_BOUNDS_S,
    tof_is_relative=True,
    constraints=[constraints.state(departure_ric, where="Front")],
)

transfer = Phase(
    name="safety_ellipse_transfer",
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    previous=initial_coast,
    link=links.impulsive(),
    final_state=arrival_ric,
    tof_bounds_s=TRANSFER_TIME_BOUNDS_S,
    tof_is_relative=True,
    constraints=[
        constraints.state(arrival_ric, where="Back"),
        constraints.solar_phase_angle(
            min_angle_deg=0.0,
            max_angle_deg=180.0,
        ),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)
mission = Mission(
    name="Composable: safety-ellipse ROE transfer",
    phases=[initial_coast, transfer],
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
phase_segments = solution.result.info["phase_segments"]
selected_initial_coast_s = (
    float(phase_segments[0]["t_end_s"])
    - float(phase_segments[0]["t_start_s"])
)
selected_transfer_s = (
    float(phase_segments[1]["t_end_s"])
    - float(phase_segments[1]["t_start_s"])
)
print(f"Selected initial coast: {selected_initial_coast_s:.6f} s")
print(f"Selected transfer duration: {selected_transfer_s:.6f} s")
print(f"Internal state layout: {solution.result.info['state_layouts'][0]}")
print(f"Constraint report: {solution.result.info['constraint_report']}")

# These two coasts are analysis-only: they visualize the natural motion before
# the departure impulse and after the arrival impulse without adding phases or
# decision variables to the optimization problem.
mission_end_s = float(solution.result.tf_s())
solved_chief_history = solution.chief_trajectory_eci
solved_deputy_history = solution.deputy_trajectory_eci
if solved_chief_history.size == 0 or solved_deputy_history.size == 0:
    raise RuntimeError("Coupled relative solution did not include absolute histories.")

solved_initial_chief = state(
    solved_chief_history[0, 0:3],
    solved_chief_history[0, 3:6],
)
solved_initial_deputy = state(
    solved_deputy_history[0, 0:3],
    solved_deputy_history[0, 3:6],
)
solved_final_chief = state(
    solved_chief_history[-1, 0:3],
    solved_chief_history[-1, 3:6],
)
terminal_maneuver = solution.result.maneuvers[-1]
if terminal_maneuver.name != "Δv (terminal)":
    raise RuntimeError("Expected the final maneuver to be the terminal impulse.")
solved_post_burn_ric = state(
    solution.traj[-1, 0:3],
    solution.traj[-1, 3:6] + terminal_maneuver.dv_mps,
)

pre_coast_times_s = np.linspace(-PRE_COAST_DURATION_S, 0.0, 121)
post_coast_elapsed_s = np.linspace(0.0, POST_COAST_DURATION_S, 121)
pre_coast = propagate_relative_numerical(
    solved_initial_chief,
    None,
    pre_coast_times_s,
    deputy_initial_eci=solved_initial_deputy,
    perturbations=FORCE_MODEL,
    initial_epoch=INITIAL_EPOCH,
    max_step_s=30.0,
    ephemeris_step_s=300.0,
)
post_coast = propagate_relative_numerical(
    solved_final_chief,
    solved_post_burn_ric,
    post_coast_elapsed_s,
    perturbations=FORCE_MODEL,
    initial_epoch=INITIAL_EPOCH + timedelta(seconds=mission_end_s),
    max_step_s=30.0,
    ephemeris_step_s=300.0,
)
pre_coast_traj = pre_coast.relative_trajectory_ric
post_coast_traj = post_coast.relative_trajectory_ric

# Shift the zero-based histories onto one mission-elapsed plot axis beginning
# at the start of the pre-transfer coast.
pre_coast_traj[:, 6] += PRE_COAST_DURATION_S
transfer_plot_traj = solution.traj.copy()
transfer_plot_traj[:, 6] += PRE_COAST_DURATION_S
post_coast_traj[:, 6] += PRE_COAST_DURATION_S + mission_end_s

# Drop repeated phase-boundary samples before stitching the plot history.
plot_traj = np.vstack(
    [
        pre_coast_traj[:-1],
        transfer_plot_traj,
        post_coast_traj[1:],
    ]
)
plot_phase_segments = [
    {
        "name": "Pre-mission coast from solved initial state",
        "t_start_s": 0.0,
        "t_end_s": PRE_COAST_DURATION_S,
        "color": "#636EFA",
    },
    *[
        {
            **segment,
            "t_start_s": PRE_COAST_DURATION_S
            + float(segment["t_start_s"]),
            "t_end_s": PRE_COAST_DURATION_S
            + float(segment["t_end_s"]),
        }
        for segment in phase_segments
    ],
    {
        "name": "Post-burn coast from solved terminal state",
        "t_start_s": PRE_COAST_DURATION_S + mission_end_s,
        "t_end_s": (
            PRE_COAST_DURATION_S
            + mission_end_s
            + POST_COAST_DURATION_S
        ),
        "color": "#00CC96",
    },
]
plot_maneuvers = [
    Maneuver(
        r_m=maneuver.r_m,
        t_s=PRE_COAST_DURATION_S + maneuver.t_s,
        dv_mps=maneuver.dv_mps,
        name=maneuver.name,
    )
    for maneuver in solution.result.maneuvers
]

save_relative_trajectory_html(
    plot_traj,
    "traj_safety_ellipse_transfer.html",
    maneuvers=plot_maneuvers,
    phase_segments=plot_phase_segments,
    title="Transfer between D'Amico safety ellipses",
)
save_trajectory_diagnostics_html(
    plot_traj,
    "diagnostics_safety_ellipse_transfer.html",
    frame_kind="relative",
    title="Safety-ellipse transfer state history",
)
