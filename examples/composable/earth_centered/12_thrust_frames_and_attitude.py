"""Earth-centered example 12: RIC-referenced kinematic thrust attitude.

Run:
  python examples/composable/earth_centered/12_thrust_frames_and_attitude.py
"""

from __future__ import annotations

import math

import numpy as np

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    ThrustControl,
    Thruster,
    constraints,
    objectives,
    state,
)
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

radius_m = 7_000_000.0
speed_mps = float(np.sqrt(EARTH.mu_m3ps2 / radius_m))
transfer_angle_rad = math.pi / 3.0
transfer_time_s = transfer_angle_rad / np.sqrt(EARTH.mu_m3ps2 / radius_m**3)

initial_state = state(
    [radius_m, 0.0, 0.0],
    [0.0, speed_mps, 0.0],
)
target_state = state(
    [
        radius_m * math.cos(transfer_angle_rad),
        radius_m * math.sin(transfer_angle_rad),
        0.0,
    ],
    [
        -(speed_mps + 0.5) * math.sin(transfer_angle_rad),
        (speed_mps + 0.5) * math.cos(transfer_angle_rad),
        0.0,
    ],
)

spacecraft = Spacecraft(
    name="Kinematic-attitude demonstrator",
    dry_mass_kg=100.0,
    thrusters=[
        Thruster(
            name="main",
            thrust_N=10.0,
            isp_s=300.0,
            propellant_mass_kg=20.0,
        )
    ],
)

# These are the three supported control representations. A mission can swap
# them without changing its dynamics or phase topology. This example solves
# the Euler representation so its angle and slew-rate states are visible.
free_vector_control = ThrustControl.vector(frame="ric")
constant_in_track_control = ThrustControl.fixed([0.0, 1.0, 0.0], frame="ric")

# Yaw=90 degrees points body +X along +I in the RIC frame. The optimizer may
# slew this attitude while respecting both path-angle and total-rate bounds.
attitude_control = ThrustControl.euler(
    frame="ric",
    initial_angles_rad=np.deg2rad([90.0, 0.0, 0.0]),
    max_slew_rate_radps=float(np.deg2rad(0.5)),
    pitch_bounds_rad=tuple(np.deg2rad([-80.0, 80.0])),
)

burn = Phase(
    name="ric_attitude_burn",
    mode="finite_thrust",
    spacecraft=spacecraft,
    dynamics=Dynamics.for_body(EARTH),
    initial_state=initial_state,
    final_state=target_state,
    tof_bounds_s=(transfer_time_s - 1.0, transfer_time_s + 1.0),
    constraints=[
        constraints.state(initial_state, where="Front"),
        constraints.state(target_state, where="Back"),
    ],
    thrust_control=attitude_control,
)

mission = Mission(
    name="Composable: RIC-referenced kinematic thrust attitude",
    phases=[burn],
    objectives=[objectives.minimize_propellant()],
    mesh_nsegs_transfer=12,
    lambert_grid_size=8,
    nrevs_to_try=(0,),
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=3,
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The attitude-controlled mission did not return a result.")

print(solution.result.summary())
print(f"Alternative free-vector control: {free_vector_control.to_dict()}")
print(f"Alternative fixed-direction control: {constant_in_track_control.to_dict()}")
if solution.attitude_phase_trajectories:
    attitude_history = solution.attitude_phase_trajectories[0]
    max_slew_degps = float(np.rad2deg(np.max(np.linalg.norm(attitude_history[:, 4:7], axis=1))))
    print(f"Final yaw/pitch/roll: {np.rad2deg(attitude_history[-1, 0:3])} deg")
    print(f"Maximum solved Euler-rate magnitude: {max_slew_degps:.6f} deg/s")

output_path = "traj_composable_thrust_frames_and_attitude.html"
save_trajectory_html(
    solution.result.traj,
    output_path,
    phase_segments=solution.result.info.get("phase_segments", []),
    title=mission.name,
)
print(f"Wrote: {output_path}")
