"""Cislunar example 27: select a periodic orbit by Jacobi constant.

The preceding periodic-orbit example fixes the initial x coordinate to choose
one member of the L1 planar Lyapunov family. Here the family member is selected
by its canonical Jacobi constant instead. ASSET is free to correct the initial
position, velocity, and period while front/back equality closes the orbit.

Run:
  python examples/composable/cislunar/27_jacobi_targeted_periodic_orbit.py
"""

from __future__ import annotations

import numpy as np

from octavian import Dynamics, Mission, Phase, Spacecraft, constraints, state
from octavian.cislunar import (
    CR3BPSystem,
    dimensionalize_state,
    dimensionalize_time,
    jacobi_constant,
    nondimensionalize_state,
    propagate_cr3bp,
)
from octavian.solvers import SolverOptions
from octavian.viz import (
    save_cr3bp_trajectory_html,
    save_trajectory_diagnostics_html,
)

system = CR3BPSystem.earth_moon()

# This nearby known solution initializes the collocation mesh. The target
# Jacobi value below, rather than this seed's x coordinate, selects the solved
# member of the planar Lyapunov family.
seed_state_canonical = state(
    [0.82, 0.0, 0.0],
    [0.0, 0.16221305707437475, 0.0],
)
seed_period_tu = 2.779749966597294
target_jacobi_canonical = 3.16

seed_state_si = dimensionalize_state(seed_state_canonical, system)
seed_period_s = float(dimensionalize_time(seed_period_tu, system))
seed_history_si = propagate_cr3bp(
    seed_state_si,
    [0.0, seed_period_s],
    system=system,
    max_step=300.0,
)
seed_terminal_si = state(
    seed_history_si[-1, 0:3],
    seed_history_si[-1, 3:6],
)

periodic_orbit = Phase(
    name="Jacobi_targeted_L1_Lyapunov",
    mode="coast",
    spacecraft=Spacecraft(name="Cislunar explorer", dry_mass_kg=250.0),
    dynamics=Dynamics.cr3bp(),
    initial_state=seed_state_si,
    final_state=seed_terminal_si,
    tof_bounds_s=(0.85 * seed_period_s, 1.15 * seed_period_s),
    constraints=[
        constraints.periodic_state(),
        # A symmetry-plane crossing removes the arbitrary phase shift around
        # the autonomous periodic orbit.
        constraints.state_component("y", 0.0, where="Front"),
        # Explicit dimensional=False keeps the target in the canonical units
        # normally used in CR3BP orbit-family tables.
        constraints.jacobi_constant(
            target_jacobi_canonical,
            where="Front",
            dimensional=False,
        ),
    ],
)

mission = Mission(
    name="Jacobi-targeted Earth-Moon L1 periodic orbit",
    phases=[periodic_orbit],
    objectives=[],
    mesh_nsegs_transfer=60,
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=5,
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if not solution.ok or solution.result is None:
    raise RuntimeError("The Jacobi-targeted periodic-orbit solve did not converge.")

trajectory_canonical = np.empty_like(solution.traj)
for row_index, row in enumerate(solution.traj):
    canonical_state = nondimensionalize_state(
        state(row[0:3], row[3:6]),
        system,
    )
    trajectory_canonical[row_index, 0:6] = np.hstack(
        [canonical_state.r_m, canonical_state.v_mps]
    )
trajectory_canonical[:, 6] = solution.traj[:, 6] / system.time_scale_s

solved_jacobi = jacobi_constant(
    trajectory_canonical[0, 0:6],
    system=system,
    dimensional=False,
)
closure_error = float(
    np.linalg.norm(trajectory_canonical[-1, 0:6] - trajectory_canonical[0, 0:6])
)

print(solution.summary())
print(f"Target canonical Jacobi constant: {target_jacobi_canonical:.12f}")
print(f"Solved canonical Jacobi constant: {solved_jacobi:.12f}")
print(f"Solved initial x: {trajectory_canonical[0, 0]:.12f} DU")
print(f"Solved period: {trajectory_canonical[-1, 6]:.12f} TU")
print(f"Canonical closure error: {closure_error:.6e}")

save_cr3bp_trajectory_html(
    trajectory_canonical,
    "traj_jacobi_targeted_L1_periodic_orbit.html",
    system=system,
    dimensional=False,
    title=mission.name,
)
save_trajectory_diagnostics_html(
    solution.traj,
    "diagnostics_jacobi_targeted_L1_periodic_orbit.html",
    frame_kind="rotating",
    cr3bp_system=system,
    title=f"{mission.name} — dimensional diagnostics",
)
