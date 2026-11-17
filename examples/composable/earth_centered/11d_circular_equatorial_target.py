"""Example 11d: exact circular/equatorial targeting and explicit alternatives.

Run: python examples/composable/earth_centered/11d_circular_equatorial_target.py
"""

from dataclasses import replace

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
    state,
)
from octavian.solvers import SolverOptions

MU = 3.986004418e14
INITIAL_EPOCH = "2026-01-01T00:00:00Z"
INITIAL_RADIUS_M = 7_000_000.0
TARGET_RADIUS_M = 7_200_000.0
TARGET_INCLINATION_DEG = 0.0

spacecraft = Spacecraft(
    name="100 mN electric vehicle",
    dry_mass_kg=20.0,
    thrusters=[Thruster(thrust_N=0.1, isp_s=1800.0, propellant_mass_kg=3.0)],
)
initial_tilt = np.deg2rad(1.0)
initial_speed = np.sqrt(MU / INITIAL_RADIUS_M)
initial = state(
    [INITIAL_RADIUS_M, 0, 0],
    [0, initial_speed * np.cos(initial_tilt), initial_speed * np.sin(initial_tilt)],
)
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
    tof_bounds_s=(24 * 3600.0, 24.2 * 3600.0),
    initial_guess=guesses.low_thrust_qlaw(throttle=0.85, plane_tol_deg=0.05),
    constraints=[
        constraints.state(initial, where="Front"),
        constraints.min_radius(6_900_000),
        constraints.semi_major_axis(TARGET_RADIUS_M, where="Back"),
        constraints.eccentricity(0.0, where="Back"),
        constraints.inclination_deg(TARGET_INCLINATION_DEG, where="Back"),
    ],
)
mission = Mission(
    name="Circular equatorial low-thrust targeting",
    initial_epoch=INITIAL_EPOCH,
    phases=[transfer],
    objectives=[objectives.minimize_propellant()],
    mesh_nsegs_transfer=120,
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=5,
        transcription_mode="LGL5",
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if not solution.ok:
    raise RuntimeError("Automatic circular/equatorial targeting failed.")
print(solution.summary())

# Equivalent terminal geometry written explicitly. Zero FPA alone only fixes
# an apsis; circular speed is also needed. z=vz=0 selects the equatorial plane;
# the automatic inclination constraint additionally selects prograde sense.
explicit = replace(
    transfer,
    initial_guess=solution.trajectory_guess(),
    constraints=[
        constraints.state(initial, where="Front"),
        constraints.semi_major_axis(TARGET_RADIUS_M, where="Back"),
        constraints.flight_path_angle_deg(0.0, where="Back"),
        constraints.velocity_magnitude(np.sqrt(MU / TARGET_RADIUS_M), where="Back"),
        constraints.state_component("z", 0.0, where="Back"),
        constraints.state_component("vz", 0.0, where="Back"),
    ],
)
solution = replace(mission, phases=[explicit]).solve()
if not solution.ok:
    raise RuntimeError("Explicit circular/equatorial targeting failed.")
print(solution.summary())
solution.viz().save_html("traj_circular_equatorial_target.html", color_by="throttle")
solution.viz().save_propulsion_html("propulsion_circular_equatorial_target.html")
