"""Earth-centered example 11b: a 100 mN small-satellite plane-change transfer.

Run: python examples/composable/earth_centered/11b_low_thrust_plane_change.py
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from octavian import (
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    Thruster,
    constraints,
    guesses,
    objectives,
    propagate,
    state,
)
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

started = perf_counter()

MU = 3.986004418e14
INITIAL_RADIUS_M = 7_000_000.0
TARGET_RADIUS_M = 7_200_000.0
TARGET_INCLINATION_DEG = 2.0

# A 20 kg dry bus carrying 3 kg of electric-propulsion propellant.
spacecraft = Spacecraft(
    name="Small electric plane-change vehicle",
    dry_mass_kg=20.0,
    thrusters=[Thruster(name="main", thrust_N=0.1, isp_s=1800.0, propellant_mass_kg=3.0)],
)
initial = state([INITIAL_RADIUS_M, 0.0, 0.0], [0.0, np.sqrt(MU / INITIAL_RADIUS_M), 0.0])
inclination = np.deg2rad(TARGET_INCLINATION_DEG)
target_speed = np.sqrt(MU / TARGET_RADIUS_M)
target = state(
    [TARGET_RADIUS_M, 0.0, 0.0],
    [0.0, target_speed * np.cos(inclination), target_speed * np.sin(inclination)],
)

transfer = Phase(
    name="electric_raise_and_plane_change",
    mode="low_thrust",
    spacecraft=spacecraft,
    dynamics=Dynamics(mu_m3ps2=MU),
    initial_state=initial,
    # Defines the Q-law target orbit, not a fixed arrival position.
    final_state=target,
    # A scheduled arrival window keeps this small-satellite tutorial quick.
    # Wide free-time, many-revolution searches are substantially larger solves.
    tof_bounds_s=(32.0 * 3600.0, 32.2 * 3600.0),
    initial_guess=guesses.low_thrust_qlaw(throttle=0.85, plane_tol_deg=0.05),
    constraints=[
        constraints.state(initial, where="Front"),
        constraints.min_radius(6_900_000.0),
        constraints.semi_major_axis(TARGET_RADIUS_M, where="Back"),
        constraints.eccentricity(0.005, where="Back", tol=0.0049),
        # Arrival phase and ascending node remain free in the optimization.
        constraints.inclination_deg(TARGET_INCLINATION_DEG, where="Back"),
    ],
)
mission = Mission(
    name="Low-thrust orbit raise with inclination change",
    phases=[transfer],
    objectives=[objectives.minimize_propellant()],
    mesh_nsegs_transfer=120,
    solver_options=SolverOptions(
        max_ls_iters=5,
        transcription_mode="LGL5",
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)
# The fuel objective is retained. A successful feasibility recovery satisfies
# the constraints; optimizer convergence is needed to establish a local optimum.
solution = mission.solve()
if not solution.ok or solution.result is None:
    raise RuntimeError("The low-thrust plane-change mission did not converge.")
print(f"Solve runtime: {perf_counter() - started:.2f} s")
print(solution.result.summary())
print("Q-law seed:", solution.result.info["phase_guess_info"][0])
for row in solution.result.info["constraint_report"]:
    print(row)

references = []
for name, orbit_state, radius, color in (
    ("Departure orbit", initial, INITIAL_RADIUS_M, "#F59E0B"),
    ("Target orbit", target, TARGET_RADIUS_M, "#C084FC"),
):
    period = 2.0 * np.pi * np.sqrt(radius**3 / MU)
    references.append(
        {
            "name": name,
            "traj": propagate.inertial(orbit_state, np.linspace(0, period, 361)),
            "color": color,
        }
    )

# Both views use actual solved controls, including coast intervals inside a phase.
for coloring in ("throttle", "firing"):
    output = f"traj_low_thrust_plane_change_{coloring}.html"
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
solution.viz().save_diagnostics_html("diagnostics_low_thrust_plane_change.html", title=mission.name)
