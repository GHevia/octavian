"""Cislunar example 27: a dimensional Earth-Moon CR3BP synodic arc.

Run:
  python examples/composable/cislunar/27_earth_moon_cr3bp.py
"""

from __future__ import annotations

import numpy as np

from octavian import state
from octavian.cislunar import (
    CR3BPSystem,
    jacobi_constant,
    propagate_cr3bp,
    synodic_to_inertial_state,
)
from octavian.viz.plotly import save_cr3bp_trajectory_html

system = CR3BPSystem.earth_moon()
lagrange_points = system.lagrange_points()
initial_position_m = lagrange_points["L1"].copy()
initial_position_m[0] += 100_000.0
initial_state = state(initial_position_m, [0.0, 0.0, 0.0])

duration_s = 10 * 12.0 * 3_600.0
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


jacobi_values = np.asarray([jacobi_constant(row[0:6], system=system) for row in reference_history])
terminal_earth_inertial = synodic_to_inertial_state(
    target_state,
    time_s=duration_s,
    system=system,
    origin="earth",
)

print(f"Earth-Moon mass parameter: {system.mass_parameter:.10f}")
print(f"Earth-Moon CR3BP period: {system.period_s / 86_400.0:.6f} days")
print(f"Propagation duration: {reference_history[-1][6] / 86_400.0:.6f} days")
print(f"Jacobi peak-to-peak drift: {np.ptp(jacobi_values):.6e} m^2/s^2")
print(f"Terminal Earth-centered inertial position: {terminal_earth_inertial.r_m / 1_000.0} km")

output_path = "traj_composable_earth_moon_cr3bp.html"
save_cr3bp_trajectory_html(
    reference_history,
    output_path,
    system=system,
    lagrange_point_names=("L1","L2", "L3", "L4", "L5"),
    title="CR3BP propagation near Earth-Moon L1",
)
print(f"Wrote: {output_path}")
