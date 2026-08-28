"""Cislunar example 29: coast-burn-transfer-burn-coast from L1 to L2.

The departure and arrival states are corrected planar Lyapunov-orbit crossings
in canonical units. The composable mission propagates along the L1 orbit,
chooses an impulsive departure into a free-time CR3BP transfer, inserts a
second impulse onto the L2 orbit, and then propagates an arrival coast.

Run:
  python examples/composable/cislunar/29_periodic_orbit_transfer.py
"""

from __future__ import annotations

import numpy as np

from octavian import (
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    constraints,
    links,
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
from octavian.solvers import SolverOptions
from octavian.viz import (
    save_cr3bp_trajectory_html,
    save_trajectory_diagnostics_html,
)

system = CR3BPSystem.earth_moon()
spacecraft = Spacecraft(name="Libration-point transfer vehicle", dry_mass_kg=400.0)
dynamics = Dynamics.cr3bp()

l1_state_canonical = state(
    [0.82, 0.0, 0.0],
    [0.0, 0.16221305707437475, 0.0],
)
l1_period_tu = 2.779749966597294
l2_state_canonical = state(
    [1.12, 0.0, 0.0],
    [0.0, 0.17796367177690003, 0.0],
)
l2_period_tu = 3.4165118813404325

l1_state_si = dimensionalize_state(l1_state_canonical, system)
l2_state_si = dimensionalize_state(l2_state_canonical, system)


def time_bounds_tu(lower_tu: float, upper_tu: float) -> tuple[float, float]:
    """Convert one canonical duration interval to seconds."""
    return (
        float(dimensionalize_time(lower_tu, system)),
        float(dimensionalize_time(upper_tu, system)),
    )


departure_coast = Phase(
    name="coast_on_L1_orbit",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    initial_state=l1_state_si,
    tof_bounds_s=time_bounds_tu(0.12, 0.18),
    tof_is_relative=True,
    constraints=[constraints.state(l1_state_si, where="Front")],
)

transfer = Phase(
    name="L1_to_L2_transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=departure_coast,
    link=links.impulsive(name="L1_departure"),
    final_state=l2_state_si,
    tof_bounds_s=time_bounds_tu(0.8, 3.0),
    tof_is_relative=True,
    variables=[variables.ImpulsiveDeltaV(where="Front")],
)

arrival_coast = Phase(
    name="coast_on_L2_orbit",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=transfer,
    link=links.impulsive(name="L2_insertion"),
    tof_bounds_s=time_bounds_tu(0.15, 0.35),
    tof_is_relative=True,
    constraints=[constraints.state(l2_state_si, where="Front")],
    variables=[variables.ImpulsiveDeltaV(where="Front")],
)

mission = Mission(
    name="Composable L1-to-L2 periodic-orbit transfer",
    phases=[departure_coast, transfer, arrival_coast],
    objectives=[objectives.minimize_total_delta_v()],
    mesh_nsegs_transfer=45,
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=5,
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The periodic-orbit transfer did not return a result.")

# Propagated reference orbits make it clear where the optimized transfer starts
# and ends. They are analysis curves, not extra optimization phases.
l1_reference = propagate_cr3bp(
    l1_state_si,
    np.linspace(0.0, float(dimensionalize_time(l1_period_tu, system)), 241),
    system=system,
    max_step=300.0,
)
l2_reference = propagate_cr3bp(
    l2_state_si,
    np.linspace(0.0, float(dimensionalize_time(l2_period_tu, system)), 241),
    system=system,
    max_step=300.0,
)

segments = [
    {
        **segment,
        "color": ("#4C78A8", "#F2CF5B", "#B279A2")[segment_index % 3],
    }
    for segment_index, segment in enumerate(solution.result.info["phase_segments"])
]
total_delta_v_mps = float(
    sum(np.linalg.norm(maneuver.dv_mps) for maneuver in solution.result.maneuvers)
)

print(solution.summary())
print(f"Transfer duration: {solution.traj[-1, 6] / system.time_scale_s:.9f} TU")
print(f"Total impulsive delta-v: {total_delta_v_mps:.6f} m/s")
for maneuver in solution.result.maneuvers:
    print(f"  {maneuver.name}: {np.linalg.norm(maneuver.dv_mps):.6f} m/s")

save_cr3bp_trajectory_html(
    solution.traj,
    "traj_L1_to_L2_periodic_orbits.html",
    system=system,
    maneuvers=solution.result.maneuvers,
    phase_segments=segments,
    reference_trajectories=[
        {"name": "L1 reference orbit", "traj": l1_reference, "color": "#54A24B"},
        {"name": "L2 reference orbit", "traj": l2_reference, "color": "#E45756"},
    ],
    title=mission.name,
)
save_trajectory_diagnostics_html(
    solution.traj,
    "diagnostics_L1_to_L2_periodic_orbits.html",
    frame_kind="rotating",
    cr3bp_system=system,
    title=mission.name,
)
