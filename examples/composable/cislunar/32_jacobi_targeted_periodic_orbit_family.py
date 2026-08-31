"""Cislunar example 32: continue a family of L1 Lyapunov orbits.

Each solve targets a nearby canonical Jacobi constant. The preceding solved
trajectory seeds the complete collocation mesh, a y=0 phase condition keeps
the same symmetry crossing, and narrow period bounds prevent convergence to a
multiple-period solution or a different periodic-orbit family.

Run:
  python examples/composable/cislunar/32_jacobi_targeted_periodic_orbit_family.py
"""

from __future__ import annotations

import numpy as np

from octavian import (
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    constraints,
    guesses,
    state,
)
from octavian.cislunar import CR3BPSystem, jacobi_constant, propagate_cr3bp
from octavian.solvers import SolverOptions
from octavian.viz import save_cr3bp_trajectory_html

system = CR3BPSystem.earth_moon()
spacecraft = Spacecraft(name="Cislunar explorer", dry_mass_kg=250.0)
dynamics = Dynamics.cr3bp(dimensional=False)

seed_state_canonical = state(
    [0.82, 0.0, 0.0],
    [0.0, 0.16221305707437475, 0.0],
)
seed_period_tu = 2.779749966597294
seed_trajectory = propagate_cr3bp(
    seed_state_canonical,
    np.linspace(0.0, seed_period_tu, 61),
    system=system,
    dimensional=False,
)


def solve_family_member(
    target_jacobi: float,
    trajectory_seed: np.ndarray,
):
    """Correct one nearby family member from the prior solved trajectory."""
    seed_period = float(trajectory_seed[-1, 6] - trajectory_seed[0, 6])
    initial_seed = state(trajectory_seed[0, 0:3], trajectory_seed[0, 3:6])
    terminal_seed = state(trajectory_seed[-1, 0:3], trajectory_seed[-1, 3:6])
    periodic_orbit = Phase(
        name="Jacobi_targeted_L1_Lyapunov",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        initial_state=initial_seed,
        final_state=terminal_seed,
        initial_guess=guesses.trajectory(trajectory_seed),
        tof_bounds_s=(0.98 * seed_period, 1.02 * seed_period),
        constraints=[
            constraints.periodic_state(),
            # Autonomous periodic orbits have an arbitrary phase. Retaining
            # this crossing condition is essential for regular continuation.
            constraints.state_component("y", 0.0, where="Front"),
            constraints.jacobi_constant(
                target_jacobi,
                where="Front",
                dimensional=False,
            ),
        ],
    )
    mission = Mission(
        name="Jacobi-targeted Earth-Moon L1 Lyapunov family",
        phases=[periodic_orbit],
        objectives=[],
        mesh_nsegs_transfer=60,
        solver_options=SolverOptions(
            print_level=3,
            max_ls_iters=5,
            enable_adaptive_mesh=False,
            asset_threads=(1, 1),
        ),
    )
    solution = mission.solve()
    if not solution.ok or solution.result is None:
        raise RuntimeError(
            f"The periodic-orbit continuation failed at C = {target_jacobi:.12f}.\n"
            f"{solution.summary()}"
        )

    trajectory = solution.traj
    solved_jacobi = jacobi_constant(
        trajectory[0, 0:6],
        system=system,
        dimensional=False,
    )
    closure_error = float(np.linalg.norm(trajectory[-1, 0:6] - trajectory[0, 0:6]))
    if abs(solved_jacobi - target_jacobi) > 1.0e-8 or closure_error > 1.0e-7:
        raise RuntimeError(
            f"Continuation returned an invalid member at C = {target_jacobi:.12f}: "
            f"solved C = {solved_jacobi:.12f}, closure = {closure_error:.3e}."
        )
    print(
        f"C = {solved_jacobi:.6f} | x0 = {trajectory[0, 0]:.9f} DU | "
        f"period = {trajectory[-1, 6]:.9f} TU | closure = {closure_error:.2e}"
    )
    return mission, trajectory, solved_jacobi


jacobi_targets = np.linspace(3.16, 3.04, 121)
print_targets = [3.16, 3.15, 3.14, 3.13, 3.12, 3.11, 3.10, 3.09, 3.08, 3.07, 3.06, 3.05, 3.04]
family: list[dict[str, object]] = []
last_mission = None
for target in jacobi_targets:
    last_mission, seed_trajectory, solved_target = solve_family_member(
        float(target),
        seed_trajectory,
    )
    if target in print_targets:
        family.append(
            {
                "name": f"L1 Lyapunov (C = {solved_target:.4f})",
                "traj": seed_trajectory.copy(),
                "color": "#E45756",
            }
        )

save_cr3bp_trajectory_html(
    np.asarray(family[0]["traj"]),
    "traj_L1_periodic_orbit_family.html",
    system=system,
    dimensional=False,
    lagrange_point_names=("L1",),
    title=last_mission.name if last_mission is not None else "L1 Lyapunov family",
    reference_trajectories=family[1:],
)
