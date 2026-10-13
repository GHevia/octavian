"""Earth-centered example 14: propagate with 8×8 Earth gravity, Sun, and Moon.

Run:
  python examples/composable/earth_centered/14_high_fidelity_propagation.py

This unpowered orbit needs no mission or optimization problem. The Earth
field uses uniform Z rotation; choose its prime-meridian angle explicitly.
"""

from __future__ import annotations

import numpy as np

from octavian import Perturbations, SphericalHarmonics, propagate, state
from octavian.viz.plotly import save_trajectory_html

initial_epoch = "2026-01-01T00:00:00Z"
duration_s = 6.0 * 3600.0
times_s = np.linspace(0.0, duration_s, 721)
initial_state = state([7_000e3, 0.0, 200e3], [0.0, 7_400.0, 800.0])

# This angle defines the coefficient frame at elapsed time zero. The epoch
# sets Sun/Moon ephemerides independently; it does not set Earth orientation.
gravity = SphericalHarmonics.earth(degree=8, order=8, reference_angle_rad=0.0)
perturbations = Perturbations(spherical_harmonics=gravity, sun=True, moon=True)
trajectory = propagate.inertial(
    initial_state,
    times_s,
    initial_epoch=initial_epoch,
    perturbations=perturbations,
    max_step_s=10.0,
    ephemeris_step_s=600.0,
)

output_path = "traj_composable_high_fidelity_propagation.html"
save_trajectory_html(
    trajectory,
    output_path,
    title="8×8 Earth gravity with Sun and Moon",
)
print(f"Propagated {duration_s / 3600.0:g} hours; final position [km]: {trajectory[-1, :3] / 1e3}")
print(f"Wrote: {output_path}")
