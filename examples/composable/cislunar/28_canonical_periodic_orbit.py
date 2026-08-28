"""Cislunar example 28: solve a canonical L1 planar Lyapunov orbit.

CR3BP literature normally publishes states in distance units (DU), velocity
units (VU), and time units (TU). ``Dynamics.cr3bp(dimensional=False)`` keeps
those canonical variables all the way through the ASSET solve and the
frame-aware plotting helpers.

Run:
  python examples/composable/cislunar/28_canonical_periodic_orbit.py
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

# A conventional canonical seed near the Earth-Moon L1 point. The seed only
# initializes the collocation phase; the constraints below define the solved
# orbit. Fixing x selects one family member and y=0 supplies a phase condition.
seed_state_canonical = state(
    [0.82, 0.0, 0.0],
    [0.0, 0.16221305707437475, 0.0],
)
seed_period_tu = 2.779749966597294

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
    name="L1_planar_Lyapunov",
    mode="coast",
    spacecraft=Spacecraft(name="Cislunar explorer", dry_mass_kg=250.0),
    dynamics=Dynamics.cr3bp(dimensional=False),
    initial_state=seed_state_canonical,
    # The propagated terminal seed supplies sensible solver scaling and an
    # initial mesh. Periodicity itself is still imposed by periodic_state().
    final_state=seed_terminal_canonical,
    # Canonical CR3BP phases interpret these values as time units (TU).
    tof_bounds_s=(0.98 * seed_period_tu, 1.02 * seed_period_tu),
    constraints=[
        # This is an ASSET FrontAndBack equality in the phase's synodic frame.
        constraints.periodic_state(),
        constraints.state_component("x", seed_state_canonical.r_m[0], where="Front"),
        constraints.state_component("y", 0.0, where="Front"),
    ],
)

mission = Mission(
    name="Canonical Earth-Moon L1 periodic orbit",
    phases=[periodic_orbit],
    objectives=[],
    mesh_nsegs_transfer=50,
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=3,
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The periodic-orbit mission did not return a result.")

# The selected dynamics keep the public solution in canonical units.
trajectory_canonical = solution.traj

closure_error = float(np.linalg.norm(trajectory_canonical[-1, 0:6] - trajectory_canonical[0, 0:6]))
jacobi_values = np.asarray(
    [
        jacobi_constant(
            row[0:6],
            system=system,
            dimensional=False,
        )
        for row in trajectory_canonical
    ]
)

print(solution.summary())
print(f"Solved period: {trajectory_canonical[-1, 6]:.12f} TU")
print(f"Canonical closure error: {closure_error:.6e}")
print(f"Canonical Jacobi peak-to-peak drift: {np.ptp(jacobi_values):.6e}")

save_cr3bp_trajectory_html(
    trajectory_canonical,
    "traj_canonical_L1_periodic_orbit.html",
    system=system,
    dimensional=False,
    title=mission.name,
)
save_trajectory_diagnostics_html(
    trajectory_canonical,
    "diagnostics_canonical_L1_periodic_orbit.html",
    frame_kind="rotating",
    cr3bp_system=system,
    cr3bp_dimensional=False,
    title=f"{mission.name} — canonical diagnostics",
)
