"""Cislunar example 27: a dimensional Earth-Moon CR3BP synodic arc.

Run:
  python examples/composable/cislunar/27_earth_moon_cr3bp.py
"""

from __future__ import annotations

import numpy as np

from octavian import (
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    constraints,
    state,
)
from octavian.cislunar import (
    CR3BPSystem,
    jacobi_constant,
    propagate_cr3bp,
    synodic_to_inertial_state,
)
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_cr3bp_trajectory_html

system = CR3BPSystem.earth_moon()
lagrange_points = system.lagrange_points()
initial_position_m = lagrange_points["L4"].copy()
initial_position_m[0] += 100_000.0
initial_state = state(initial_position_m, [0.0, 0.0, 0.0])

duration_s = 12.0 * 3_600.0
reference_history = propagate_cr3bp(
    initial_state,
    [0.0, duration_s],
    system=system,
    max_step=300.0,
)
target_state = state(
    reference_history[-1, 0:3],
    reference_history[-1, 3:6],
)

probe = Spacecraft(name="Cislunar probe", dry_mass_kg=100.0)
dynamics = Dynamics.cr3bp()
arc = Phase(
    name="earth_moon_synodic_arc",
    mode="coast",
    spacecraft=probe,
    dynamics=dynamics,
    initial_state=initial_state,
    final_state=target_state,
    tof_bounds_s=(duration_s - 1.0, duration_s + 1.0),
    constraints=[
        constraints.state(initial_state, where="Front"),
        constraints.state(target_state, where="Back"),
    ],
)

mission = Mission(
    name="Composable: dimensional Earth-Moon CR3BP arc",
    phases=[arc],
    mesh_nsegs_transfer=16,
    solver_options=SolverOptions(
        print_level=0,
        max_ls_iters=3,
        enable_adaptive_mesh=False,
        asset_threads=(1, 1),
    ),
)

solution = mission.solve()
if solution.result is None:
    raise RuntimeError("The CR3BP mission did not return a result.")

jacobi_values = np.asarray(
    [jacobi_constant(row[0:6], system=system) for row in solution.result.traj]
)
terminal_earth_inertial = synodic_to_inertial_state(
    target_state,
    time_s=duration_s,
    system=system,
    origin="earth",
)

print(solution.result.summary())
print(f"Earth-Moon mass parameter: {system.mass_parameter:.10f}")
print(f"Earth-Moon CR3BP period: {system.period_s / 86_400.0:.6f} days")
print(f"Jacobi peak-to-peak drift: {np.ptp(jacobi_values):.6e} m^2/s^2")
print(f"Terminal Earth-centered inertial position: {terminal_earth_inertial.r_m / 1_000.0} km")

output_path = "traj_composable_earth_moon_cr3bp.html"
save_cr3bp_trajectory_html(
    solution.result.traj,
    output_path,
    system=system,
    title=mission.name,
)
print(f"Wrote: {output_path}")
