"""Analysis example 1: one namespace for Octavian propagators.

The specialized functions remain available in their scientific subpackages.
For ordinary analysis scripts, ``octavian.propagate`` makes the available
models easy to discover and gives every state history a final time column.
"""

from __future__ import annotations

import numpy as np

from octavian import EARTH, propagate, state
from octavian.astro import classical_to_cartesian
from octavian.relative import RelativeOrbitalElements

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

two_body = propagate.two_body(
    chief,
    times_s,
    mu_m3ps2=EARTH.mu_m3ps2,
)
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

try:
    from octavian import CR3BPSystem
except ImportError:
    cr3bp = None
else:
    earth_moon = CR3BPSystem.earth_moon()
    l4 = state(
        earth_moon.lagrange_points(dimensional=False)["L4"],
        [0.0, 0.0, 0.0],
    )
    cr3bp = propagate.cr3bp(
        l4,
        [0.0, 0.01],
        system=earth_moon,
        dimensional=False,
    )

print("two-body history:", two_body.shape)
print("CWH history:", cwh.shape)
print("nonlinear RIC history:", nonlinear_ric.shape)
print("coupled relative RIC history:", coupled.relative_trajectory_ric.shape)
print("relative-element history:", relative_elements.elements.shape)
print("same propagation in RIC:", relative_elements.ric.shape)
if cr3bp is not None:
    print("CR3BP history:", cr3bp.shape)
