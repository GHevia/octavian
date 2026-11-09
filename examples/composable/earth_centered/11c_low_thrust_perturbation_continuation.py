"""Earth-centered example 11c: reuse solved low-thrust trajectories as guesses.

Run: python examples/composable/earth_centered/11c_low_thrust_perturbation_continuation.py
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from octavian import (
    Dynamics,
    Mission,
    Perturbations,
    Phase,
    Spacecraft,
    SphericalHarmonics,
    Thruster,
    constraints,
    guesses,
    objectives,
    propagate,
    state,
)
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

# Keep GM, epoch, endpoints, vehicle, and control conventions fixed at every
# stage. Only the perturbations change. The final targets are osculating
# elements at arrival, with orbital phase and ascending node left free.
gravity = SphericalHarmonics.earth(degree=8, order=8, reference_angle_rad=0.0)
MU = gravity.reference_mu_m3ps2
INITIAL_EPOCH = "2026-01-01T00:00:00Z"
INITIAL_RADIUS_M = 7_000_000.0
TARGET_RADIUS_M = 7_200_000.0
TARGET_INCLINATION_DEG = 2.0

spacecraft = Spacecraft(
    name="100 mN electric vehicle",
    dry_mass_kg=20.0,
    thrusters=[Thruster(thrust_N=0.1, isp_s=1800.0, propellant_mass_kg=3.0)],
)
initial = state([INITIAL_RADIUS_M, 0, 0], [0, np.sqrt(MU / INITIAL_RADIUS_M), 0])
inclination = np.deg2rad(TARGET_INCLINATION_DEG)
speed = np.sqrt(MU / TARGET_RADIUS_M)
target = state(
    [TARGET_RADIUS_M, 0, 0], [0, speed * np.cos(inclination), speed * np.sin(inclination)]
)
transfer = Phase(
    name="electric_raise_and_plane_change",
    mode="low_thrust",
    spacecraft=spacecraft,
    dynamics=Dynamics(mu_m3ps2=MU),
    initial_state=initial,
    final_state=target,
    tof_bounds_s=(32 * 3600.0, 32.2 * 3600.0),
    initial_guess=guesses.low_thrust_qlaw(throttle=0.85, plane_tol_deg=0.05),
    constraints=[
        constraints.state(initial, where="Front"),
        constraints.min_radius(6_900_000),
        constraints.semi_major_axis(TARGET_RADIUS_M, where="Back"),
        constraints.eccentricity(0.005, where="Back", tol=0.0049),
        constraints.inclination_deg(TARGET_INCLINATION_DEG, where="Back"),
    ],
)
mission = Mission(
    name="Low-thrust perturbation continuation",
    initial_epoch=INITIAL_EPOCH,
    phases=[transfer],
    objectives=[objectives.minimize_propellant()],
    mesh_nsegs_transfer=160,
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=5,
        transcription_mode="LGL5",
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

stages = [
    ("two_body", Perturbations()),
    ("j2", Perturbations(j2=True)),
    ("earth_8x8", Perturbations(spherical_harmonics=gravity)),
    ("earth_8x8_sun_moon", Perturbations(spherical_harmonics=gravity, sun=True, moon=True)),
]
previous_solution = None
solutions = []
for label, perturbations in stages:
    # Q-law initializes only the first solve. Later stages reuse solved mass,
    # time, and every control sample, not just the plotting [R,V,t] trajectory.
    initial_guess = (
        transfer.initial_guess
        if previous_solution is None
        else previous_solution.trajectory_guess()
    )
    phase = replace(
        transfer,
        dynamics=Dynamics(
            mu_m3ps2=MU,
            perturbations=perturbations,
            third_body_table_step_s=600.0,
        ),
        initial_guess=initial_guess,
    )
    stage_mission = replace(mission, name=f"{mission.name}: {label}", phases=[phase])
    started = perf_counter()
    solution = stage_mission.solve()
    if not solution.ok or solution.result is None:
        raise RuntimeError(f"Low-thrust continuation failed at {label}.")
    print(f"\n{label}: {perf_counter() - started:.2f} s")
    print(solution.summary())
    print("Initial guess:", solution.result.info["phase_guess_info"][0])
    for row in solution.result.info["constraint_report"]:
        print(row)
    solutions.append(solution)
    previous_solution = solution

# These are four solves of the same transfer, not four successive flight arcs.
# A successful feasibility recovery satisfies constraints; fuel optimality
# additionally requires convergence of the optimizer's optimality conditions.
solution = solutions[-1]
references = []
for name, orbit_state, radius, color in (
    ("Departure two-body reference", initial, INITIAL_RADIUS_M, "#F59E0B"),
    ("Target two-body reference", target, TARGET_RADIUS_M, "#C084FC"),
):
    period = 2 * np.pi * np.sqrt(radius**3 / MU)
    references.append(
        {
            "name": name,
            "traj": propagate.inertial(orbit_state, np.linspace(0, period, 361)),
            "color": color,
        }
    )
for coloring in ("throttle", "firing"):
    output = f"traj_low_thrust_perturbation_continuation_{coloring}.html"
    save_trajectory_html(
        solution.traj,
        output,
        phase_segments=solution.result.info["phase_segments"],
        thrust_histories=solution.phase_thrust_trajectories,
        color_by=coloring,
        firing_threshold=0.001,
        reference_trajectories=references,
        title=mission.name,
    )
    print("Wrote:", output)
solution.viz().save_diagnostics_html(
    "diagnostics_low_thrust_perturbation_continuation.html", title=mission.name
)
