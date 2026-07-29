"""Relative composable example 16: exact RIC formulations and CWH linearization.

This analysis-only example compares the exact six-state circular-chief RIC
equations with independent chief/deputy two-body propagation. It also shows
the stacked chief-ECI/deputy-RIC model used for eccentric chief orbits.
"""

from __future__ import annotations

import numpy as np

from octavian import EARTH, Dynamics, state
from octavian.relative import (
    propagate_nonlinear_relative_ric,
    propagate_relative_numerical,
)

CHIEF_RADIUS_M = EARTH.mean_radius_m + 400_000.0
chief_eci = state(
    [CHIEF_RADIUS_M, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / CHIEF_RADIUS_M), 0.0],
)
deputy_ric = state(
    [500.0, -800.0, 100.0],
    [0.05, -0.02, 0.01],
)
times_s = np.linspace(0.0, 600.0, 61)

direct_ric = propagate_nonlinear_relative_ric(
    np.hstack([deputy_ric.r_m, deputy_ric.v_mps]),
    times_s,
    mu_m3ps2=EARTH.mu_m3ps2,
    chief_orbit_radius_m=CHIEF_RADIUS_M,
    max_step_s=1.0,
)
coupled_eci = propagate_relative_numerical(
    chief_eci,
    deputy_ric,
    times_s,
    max_step_s=1.0,
)
np.testing.assert_allclose(
    direct_ric[:, 0:6],
    coupled_eci.relative_states_ric,
    atol=1.0e-5,
)

direct_dynamics = Dynamics.relative(
    chief_initial_state_eci=chief_eci,
    propagation_mode="nonlinear_ric",
)
stacked_dynamics = Dynamics.relative(
    chief_initial_state_eci=chief_eci,
    propagation_mode="coupled_ric",
)

print(f"Direct exact state: {direct_dynamics.model.state_representation}")
print(f"Stacked exact state: {stacked_dynamics.model.state_representation}")
print(
    "Maximum direct-vs-coupled RIC difference: "
    f"{np.max(np.abs(direct_ric[:, 0:6] - coupled_eci.relative_states_ric)):.3e}"
)
