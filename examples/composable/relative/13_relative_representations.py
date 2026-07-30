"""Relative composable example 13: absolute, RIC, and element conversions.

This example has no optimizer. It demonstrates the representation layer used
at the boundary of both CWH and full nonlinear relative missions.
"""

from __future__ import annotations

import numpy as np

from octavian import EARTH, state
from octavian.astro import classic_to_cartesian
from octavian.relative import (
    absolute_to_classical_relative_orbital_elements,
    absolute_to_relative_history,
    absolute_to_relative_orbital_elements,
    absolute_to_relative_state,
    chief_ric_angular_velocity,
    classical_to_damico_relative_orbital_elements,
    damico_to_classical_relative_orbital_elements,
    relative_orbital_elements_to_relative_state,
    relative_state_to_relative_orbital_elements,
    relative_to_absolute_history,
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
classical_differences = absolute_to_classical_relative_orbital_elements(
    chief_eci,
    deputy_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
)
ric_to_damico = relative_state_to_relative_orbital_elements(
    chief_eci,
    deputy_ric,
    mu_m3ps2=EARTH.mu_m3ps2,
)
damico_to_ric = relative_orbital_elements_to_relative_state(
    chief_eci,
    relative_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)
damico_to_classical = damico_to_classical_relative_orbital_elements(
    chief_eci,
    relative_elements,
    mu_m3ps2=EARTH.mu_m3ps2,
)
classical_to_damico = classical_to_damico_relative_orbital_elements(
    chief_eci,
    classical_differences,
    mu_m3ps2=EARTH.mu_m3ps2,
)
ric_angular_velocity = chief_ric_angular_velocity(chief_eci)

# History converters accept the same seven-column state-and-time convention
# used by solver results. They vectorize the per-state transforms above.
chief_history = np.vstack(
    [
        np.hstack([chief_eci.r_m, chief_eci.v_mps, 0.0]),
        np.hstack([chief_eci.r_m, chief_eci.v_mps, 10.0]),
    ]
)
deputy_history = np.vstack(
    [
        np.hstack([deputy_eci.r_m, deputy_eci.v_mps, 0.0]),
        np.hstack([deputy_eci.r_m, deputy_eci.v_mps, 10.0]),
    ]
)
relative_history = absolute_to_relative_history(chief_history, deputy_history)
recovered_deputy_history = relative_to_absolute_history(
    chief_history,
    relative_history,
)

np.testing.assert_allclose(recovered_ric.r_m, deputy_ric.r_m, atol=1.0e-8)
np.testing.assert_allclose(recovered_ric.v_mps, deputy_ric.v_mps, atol=1.0e-10)
np.testing.assert_allclose(damico_to_ric.r_m, deputy_ric.r_m, atol=1.0e-6)
np.testing.assert_allclose(
    ric_to_damico.as_vector(),
    relative_elements.as_vector(),
    atol=1.0e-12,
)
np.testing.assert_allclose(
    damico_to_classical.as_vector(),
    classical_differences.as_vector(),
    atol=1.0e-8,
)
np.testing.assert_allclose(
    classical_to_damico.as_vector(),
    relative_elements.as_vector(),
    atol=1.0e-12,
)
np.testing.assert_allclose(
    recovered_deputy_history,
    deputy_history,
    atol=1.0e-8,
)

print("Chief ECI state:")
print(np.hstack([chief_eci.r_m, chief_eci.v_mps]))
print("Deputy ECI state:")
print(np.hstack([deputy_eci.r_m, deputy_eci.v_mps]))
print("Recovered deputy RIC state:")
print(np.hstack([recovered_ric.r_m, recovered_ric.v_mps]))
print("D'Amico quasi-nonsingular relative orbital elements:")
print(relative_elements.as_vector())
print("Classical relative orbital-element differences:")
print(classical_differences.as_vector())
print("Chief RIC angular velocity expressed in RIC:")
print(ric_angular_velocity)
