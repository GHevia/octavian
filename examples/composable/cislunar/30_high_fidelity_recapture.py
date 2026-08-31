"""Cislunar example 30: recapture a CR3BP orbit in a perturbed inertial model.

This is an explicit model handoff, not a claim that a CR3BP periodic orbit
remains periodic in an ephemeris model. A nominal L1 Lyapunov orbit is aligned
with the BSP Moon, converted from synodic to Earth-centered inertial states,
and used as the boundary reference for a second ASSET mission with Earth J2,
ephemeris Moon/Sun gravity, and cannonball SRP. Boundary impulses measure the
correction needed to recapture the nominal endpoint.

Run:
  python examples/composable/cislunar/30_high_fidelity_recapture.py
"""

from __future__ import annotations

import numpy as np

from octavian import (
    EARTH,
    Cannonball,
    Dynamics,
    Mission,
    Perturbations,
    Phase,
    Spacecraft,
    constraints,
    guesses,
    objectives,
    state,
    variables,
)
from octavian.cislunar import (
    CR3BPSystem,
    dimensionalize_state,
    dimensionalize_time,
    propagate_cr3bp,
)
from octavian.data.ephemeris import sample_sun_moon_positions_eci_tod
from octavian.solvers import SolverOptions
from octavian.viz import save_trajectory_diagnostics_html, save_trajectory_html

INITIAL_EPOCH = "2026-01-01T00:00:00Z"
system = CR3BPSystem.earth_moon()

canonical_initial = state(
    [0.82, 0.0, 0.0],
    [0.0, 0.16221305707437475, 0.0],
)
period_tu = 2.779749966597294
period_s = float(dimensionalize_time(period_tu, system))
synodic_initial = dimensionalize_state(canonical_initial, system)
synodic_history = propagate_cr3bp(
    synodic_initial,
    np.linspace(0.0, period_s, 241),
    system=system,
    max_step=300.0,
)
# Sample the same BSP geometry used by the perturbed dynamics. The handoff
# below embeds the circular solution in an ephemeris-aligned rotating/pulsating
# frame, avoiding an artificial correction caused by the Moon's inclination
# and roughly 12% peak-to-peak distance variation over this arc.
ephemeris_times_s, ephemeris_positions = sample_sun_moon_positions_eci_tod(
    initial_epoch=INITIAL_EPOCH,
    duration_s=period_s,
    step_s=3_600.0,
)
moon_positions_m = ephemeris_positions["moon"]
moon_velocities_mps = np.gradient(
    moon_positions_m,
    ephemeris_times_s,
    axis=0,
    edge_order=2,
)


def interpolate_ephemeris(samples: np.ndarray, time_s: float) -> np.ndarray:
    """Interpolate a three-component BSP history at mission-relative time."""
    return np.asarray(
        [np.interp(time_s, ephemeris_times_s, samples[:, component]) for component in range(3)]
    )


def nominal_inertial_row(synodic_row: np.ndarray) -> np.ndarray:
    """Embed one CR3BP sample in the instantaneous Earth-Moon geometry."""
    time_s = float(synodic_row[6])
    moon_position_m = interpolate_ephemeris(moon_positions_m, time_s)
    moon_velocity_mps = interpolate_ephemeris(moon_velocities_mps, time_s)
    moon_distance_m = float(np.linalg.norm(moon_position_m))

    x_axis = moon_position_m / moon_distance_m
    angular_velocity = np.cross(moon_position_m, moon_velocity_mps) / moon_distance_m**2
    z_axis = angular_velocity / np.linalg.norm(angular_velocity)
    y_axis = np.cross(z_axis, x_axis)
    rotating_basis = np.column_stack([x_axis, y_axis, z_axis])

    relative_position_canonical = (
        synodic_row[0:3] - system.primary_position_m
    ) / system.separation_m
    relative_velocity_canonical_per_s = synodic_row[3:6] / system.separation_m
    moon_radial_speed_mps = float(np.dot(x_axis, moon_velocity_mps))

    inertial_position_m = rotating_basis @ (moon_distance_m * relative_position_canonical)
    inertial_velocity_mps = np.cross(angular_velocity, inertial_position_m) + rotating_basis @ (
        moon_radial_speed_mps * relative_position_canonical
        + moon_distance_m * relative_velocity_canonical_per_s
    )
    return np.hstack([inertial_position_m, inertial_velocity_mps, time_s])


nominal_inertial_history = np.asarray([nominal_inertial_row(row) for row in synodic_history])

inertial_initial = state(
    nominal_inertial_history[0, 0:3],
    nominal_inertial_history[0, 3:6],
)
inertial_target = state(
    nominal_inertial_history[-1, 0:3],
    nominal_inertial_history[-1, 3:6],
)

spacecraft = Spacecraft(
    name="High-fidelity cislunar explorer",
    dry_mass_kg=500.0,
    cannonball=Cannonball(
        srp_area_m2=12.0,
        reflectivity_coefficient=1.4,
    ),
)
perturbed_dynamics = Dynamics.for_body(
    EARTH,
    perturbations=Perturbations(
        j2=True,
        moon=True,
        sun=True,
        srp=True,
    ),
    third_body_table_step_s=3_600.0,
)

recapture = Phase(
    name="perturbed_inertial_recapture",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=perturbed_dynamics,
    initial_state=inertial_initial,
    final_state=inertial_target,
    tof_bounds_s=(period_s - 3_600.0, period_s + 3_600.0),
    initial_guess=guesses.trajectory(nominal_inertial_history[::4]),
    constraints=[
        constraints.state(inertial_initial, where="Front"),
        constraints.state(inertial_target, where="Back"),
    ],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)

mission = Mission(
    name="CR3BP-to-ephemeris perturbed recapture",
    initial_epoch=INITIAL_EPOCH,
    phases=[recapture],
    objectives=[objectives.minimize_total_delta_v()],
    mesh_nsegs_transfer=60,
    lambert_grid_size=30,
    nrevs_to_try=(0, 1),
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=5,
        enable_adaptive_mesh=True,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The high-fidelity recapture did not return a result.")

total_delta_v_mps = float(
    sum(np.linalg.norm(maneuver.dv_mps) for maneuver in solution.result.maneuvers)
)
print(solution.summary())
print(f"Nominal CR3BP period: {period_s / 86_400.0:.6f} days")
print(f"Perturbed-model recapture delta-v: {total_delta_v_mps:.6f} m/s")
print(
    "This correction measures model mismatch; it does not make the "
    "ephemeris trajectory mathematically periodic."
)

save_trajectory_html(
    solution.traj,
    "traj_high_fidelity_cislunar_recapture.html",
    maneuvers=solution.result.maneuvers,
    phase_segments=solution.result.info["phase_segments"],
    title=mission.name,
    use_earth_texture=False,
)
save_trajectory_diagnostics_html(
    solution.traj,
    "diagnostics_high_fidelity_cislunar_recapture.html",
    frame_kind="inertial",
    mu_m3ps2=EARTH.mu_m3ps2,
    title=mission.name,
)
