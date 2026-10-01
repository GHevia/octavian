"""Analysis example 1: one namespace for Octavian propagators.

The specialized functions remain available in their scientific subpackages.
For ordinary analysis scripts, ``octavian.propagate`` makes the available
models easy to discover and gives every state history a final time column.
"""

from __future__ import annotations

import numpy as np

from octavian import EARTH, CR3BPSystem, propagate, state
from octavian.astro import classical_to_cartesian
from octavian.relative import RelativeOrbitalElements

# Define an absolute chief orbit and a deputy offset in radial/in-track/cross-track axes.
chief_position, chief_velocity = classical_to_cartesian(
    a_m=EARTH.mean_radius_m + 500_000.0,
    e=0.001,
    inc_deg=40.0,
    raan_deg=20.0,
    argp_deg=10.0,
    true_anomaly_deg=30.0,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief = state(chief_position, chief_velocity)
times_s = np.linspace(0.0, 300.0, 7)
mean_motion_radps = np.sqrt(EARTH.mu_m3ps2 / (EARTH.mean_radius_m + 500_000.0) ** 3)
relative_initial = state([100.0, -500.0, 50.0], [0.0, 0.02, 0.0])
relative_vector = np.hstack([relative_initial.r_m, relative_initial.v_mps])

# Absolute orbit: two-body gravity. Rows are [x, y, z, vx, vy, vz, time] in SI.
two_body = propagate.two_body(
    chief,
    times_s,
    mu_m3ps2=EARTH.mu_m3ps2,
)
# Relative orbit: choose a linear CWH approximation or an exact nonlinear model.
cwh = propagate.cwh(
    relative_vector,
    times_s,
    mean_motion_radps=mean_motion_radps,
)
nonlinear_ric = propagate.nonlinear_ric(
    relative_vector,
    times_s,
    mu_m3ps2=EARTH.mu_m3ps2,
    chief_orbit_radius_m=EARTH.mean_radius_m + 500_000.0,
)
coupled = propagate.relative(
    chief,
    relative_initial,
    times_s,
)

# Relative orbital elements describe the deputy through differences from the chief.
initial_roe = RelativeOrbitalElements(
    delta_a=1.0e-4,
    delta_lambda_rad=-0.002,
    delta_ex=1.0e-4,
    delta_ey=-2.0e-4,
    delta_ix_rad=3.0e-4,
    delta_iy_rad=-4.0e-4,
)
relative_elements = propagate.relative_elements(
    initial_roe,
    times_s,
    chief_initial_state_eci=chief,
    mu_m3ps2=EARTH.mu_m3ps2,
)

# CR3BP uses synodic coordinates; dimensional=False selects canonical units.
earth_moon = CR3BPSystem.earth_moon()
l4 = state(earth_moon.lagrange_points(dimensional=False)["L4"], [0.0, 0.0, 0.0])
cr3bp = propagate.cr3bp(l4, [0.0, 0.01], system=earth_moon, dimensional=False)

print("Chief position after 300 s [km]:", two_body[-1, :3] / 1e3)
print("CWH deputy offset [m]:", cwh[-1, :3])
print("Nonlinear circular-chief offset [m]:", nonlinear_ric[-1, :3])
print("Coupled chief/deputy offset [m]:", coupled.relative_trajectory_ric[-1, :3])
print("Relative elements converted to RIC [m]:", relative_elements.ric[-1, :3])
print("Earth–Moon L4 position [canonical units]:", cr3bp[-1, :3])
