"""Cislunar example 26: recapture a CR3BP orbit in a perturbed inertial model.

This is an explicit model handoff, not a claim that a CR3BP periodic orbit
remains periodic in an ephemeris model. A nominal L1 Lyapunov orbit is aligned
with the BSP Moon, converted from synodic to Earth-centered inertial states,
and used as the boundary reference for a second ASSET mission with Earth J2,
ephemeris Moon/Sun gravity, and cannonball SRP. Boundary impulses measure the
correction needed to recapture the nominal endpoint.

Run:
  python examples/composable/cislunar/26_high_fidelity_recapture.py
"""

from __future__ import annotations

import math

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
    objectives,
    state,
    variables,
)
from octavian.cislunar import (
    CR3BPSystem,
    dimensionalize_state,
    dimensionalize_time,
    propagate_cr3bp,
    synodic_to_inertial_state,
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
synodic_terminal = state(
    synodic_history[-1, 0:3],
    synodic_history[-1, 3:6],
)

# Align synodic +X with the actual Earth-to-Moon direction at the handoff
# epoch. This makes the circular CR3BP frame and the ECI_TOD perturbation
# tables geometrically consistent at t=0.
_, ephemeris_positions = sample_sun_moon_positions_eci_tod(
    initial_epoch=INITIAL_EPOCH,
    duration_s=period_s,
    step_s=6.0 * 3_600.0,
)
moon_at_epoch_m = ephemeris_positions["moon"][0]
phase_at_epoch_rad = math.atan2(moon_at_epoch_m[1], moon_at_epoch_m[0])

inertial_initial = synodic_to_inertial_state(
    synodic_initial,
    time_s=0.0,
    system=system,
    origin="earth",
    phase_at_epoch_rad=phase_at_epoch_rad,
)
inertial_target = synodic_to_inertial_state(
    synodic_terminal,
    time_s=period_s,
    system=system,
    origin="earth",
    phase_at_epoch_rad=phase_at_epoch_rad,
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
    third_body_table_step_s=6.0 * 3_600.0,
)

recapture = Phase(
    name="perturbed_inertial_recapture",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=perturbed_dynamics,
    initial_state=inertial_initial,
    final_state=inertial_target,
    tof_bounds_s=(period_s - 3_600.0, period_s + 3_600.0),
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
        enable_adaptive_mesh=False,
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
