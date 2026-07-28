"""Composable example 13: move between absolute, RIC, and relative elements.

This example has no optimizer. It demonstrates the representation layer used
at the boundary of both CWH and full nonlinear relative missions.
"""

from __future__ import annotations

import numpy as np

from octavian import EARTH, state
from octavian.astro import classic_to_cartesian
from octavian.relative import (
    absolute_to_relative_orbital_elements,
    absolute_to_relative_state,
    relative_to_absolute_state,
)

chief_position, chief_velocity = classic_to_cartesian(
    a_m=EARTH.mean_radius_m + 500_000.0,
    e=0.001,
    inc_deg=51.6,
    raan_deg=25.0,
    argp_deg=15.0,
    true_anomaly_deg=40.0,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief_eci = state(chief_position, chief_velocity)
deputy_ric = state(
    r_m=[100.0, -500.0, 25.0],
    v_mps=[0.02, 0.01, -0.005],
)

deputy_eci = relative_to_absolute_state(chief_eci, deputy_ric)
recovered_ric = absolute_to_relative_state(chief_eci, deputy_eci)
relative_elements = absolute_to_relative_orbital_elements(
    chief_eci,
    deputy_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
)

np.testing.assert_allclose(recovered_ric.r_m, deputy_ric.r_m, atol=1.0e-8)
np.testing.assert_allclose(recovered_ric.v_mps, deputy_ric.v_mps, atol=1.0e-10)

print("Chief ECI state:")
print(np.hstack([chief_eci.r_m, chief_eci.v_mps]))
print("Deputy ECI state:")
print(np.hstack([deputy_eci.r_m, deputy_eci.v_mps]))
print("Recovered deputy RIC state:")
print(np.hstack([recovered_ric.r_m, recovered_ric.v_mps]))
print("Quasi-nonsingular relative orbital elements:")
print(relative_elements.as_vector())
