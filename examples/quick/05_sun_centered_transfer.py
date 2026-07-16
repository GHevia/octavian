"""Quick example 05: Sun-centered Earth-orbit to Mars-orbit transfer.

This is an idealized heliocentric two-body example. The endpoints are circular
coplanar orbits rather than planetary ephemeris states.

Run:
  python examples/quick/05_sun_centered_transfer.py

Outputs:
  - prints a short solution summary
  - prints the frame and characteristic scaling recorded by the result
"""

from __future__ import annotations

import numpy as np

from octavian import SUN, state, two_burn_rendezvous
from octavian.solvers import SolverOptions

AU_M = 149_597_870_700.0
EARTH_ORBIT_RADIUS_M = AU_M
MARS_ORBIT_RADIUS_M = 1.523679 * AU_M

transfer_semimajor_axis_m = 0.5 * (EARTH_ORBIT_RADIUS_M + MARS_ORBIT_RADIUS_M)
hohmann_time_s = float(
    np.pi * np.sqrt(transfer_semimajor_axis_m**3 / SUN.mu_m3ps2)
)

x0 = state(
    r_m=[EARTH_ORBIT_RADIUS_M, 0.0, 0.0],
    v_mps=[0.0, np.sqrt(SUN.mu_m3ps2 / EARTH_ORBIT_RADIUS_M), 0.0],
)
xf = state(
    # An infinitesimal offset from exact opposition avoids the geometric
    # singularity of a Lambert solve with exactly collinear endpoints.
    r_m=MARS_ORBIT_RADIUS_M * np.array([np.cos(np.pi - 1e-6), np.sin(np.pi - 1e-6), 0.0]),
    v_mps=np.sqrt(SUN.mu_m3ps2 / MARS_ORBIT_RADIUS_M)
    * np.array([-np.sin(np.pi - 1e-6), np.cos(np.pi - 1e-6), 0.0]),
)

mission = two_burn_rendezvous(
    x0,
    xf,
    central_body=SUN,
    tf_bounds_s=(0.8 * hohmann_time_s, 1.2 * hohmann_time_s),
    nsegs=60,
    lambert_grid_size=60,
    nrevs_to_try=(0,),
    solver_options=SolverOptions(print_level=0),
    name="Quick: idealized Sun-centered Earth-to-Mars transfer",
)

solution = mission.solve()
print(solution.summary())
print(f"Frame: {solution.frame}")
print(f"Scaling: {solution.scaling}")
