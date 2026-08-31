"""Cislunar example 31: select a periodic orbit by Jacobi constant.

The preceding periodic-orbit example fixes the initial x coordinate to choose
one member of the L1 planar Lyapunov family. Here the family member is selected
by its canonical Jacobi constant instead. ASSET is free to correct the initial
position, velocity, and period while front/back equality closes the orbit.

Run:
  python examples/composable/cislunar/31_jacobi_targeted_periodic_orbit.py
"""

from __future__ import annotations

import numpy as np

from octavian import Dynamics, Mission, Phase, Spacecraft, constraints, state
from octavian.cislunar import (
    CR3BPSystem,
    jacobi_constant,
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

seed_history_canonical = propagate_cr3bp(
    seed_state_canonical,
    [0.0, seed_period_tu],
    system=system,
    dimensional=False,
)
seed_terminal_canonical = state(
    seed_history_canonical[-1, 0:3],
    seed_history_canonical[-1, 3:6],
)

periodic_orbit = Phase(
    name="Jacobi_targeted_L1_Lyapunov",
    mode="coast",
    spacecraft=Spacecraft(name="Cislunar explorer", dry_mass_kg=250.0),
    dynamics=Dynamics.cr3bp(dimensional=False),
    initial_state=seed_state_canonical,
    final_state=seed_terminal_canonical,
    tof_bounds_s=(0.85 * seed_period_tu, 1.15 * seed_period_tu),
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

trajectory_canonical = solution.traj

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
    lagrange_point_names=("L1",),
    title=mission.name,
)
save_trajectory_diagnostics_html(
    trajectory_canonical,
    "diagnostics_jacobi_targeted_L1_periodic_orbit.html",
    frame_kind="rotating",
    cr3bp_system=system,
    cr3bp_dimensional=False,
    title=f"{mission.name} — canonical diagnostics",
)
