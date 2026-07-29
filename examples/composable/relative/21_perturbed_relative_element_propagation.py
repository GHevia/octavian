"""Relative composable example 21: propagate D'Amico ROEs with perturbations.

The analytical history shows native two-body ROE drift. The perturbed history
uses the same initial ROEs, but converts them to an absolute deputy state and
propagates chief and deputy together under central gravity, J2, and solar
third-body gravity before reconstructing osculating ROEs and RIC states.
"""

from __future__ import annotations

import numpy as np

from octavian import EARTH, Perturbations, state
from octavian.astro import classical_to_cartesian
from octavian.relative import (
    RelativeOrbitalElements,
    propagate_relative_elements_to_ric,
    propagate_relative_orbital_elements,
)
from octavian.viz import (
    save_relative_trajectory_html,
    save_trajectory_diagnostics_html,
)

INITIAL_EPOCH = "2026-01-01T00:00:00Z"
CHIEF_SEMI_MAJOR_AXIS_M = 7_000_000.0
chief_position, chief_velocity = classical_to_cartesian(
    a_m=CHIEF_SEMI_MAJOR_AXIS_M,
    e=0.001,
    inc_deg=40.0,
    raan_deg=20.0,
    argp_deg=10.0,
    true_anomaly_deg=30.0,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief_eci = state(chief_position, chief_velocity)
initial_elements = RelativeOrbitalElements(
    delta_a=200.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_lambda_rad=-3_000.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ex=500.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ey=-300.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ix_rad=400.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_iy_rad=-200.0 / CHIEF_SEMI_MAJOR_AXIS_M,
)
times_s = np.linspace(0.0, 6.0 * 3_600.0, 145)
force_model = Perturbations(j2=True, sun=True)

two_body_elements = propagate_relative_orbital_elements(
    initial_elements,
    times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
)
perturbed_elements = propagate_relative_orbital_elements(
    initial_elements,
    times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=force_model,
    initial_epoch=INITIAL_EPOCH,
    max_step_s=30.0,
    ephemeris_step_s=300.0,
)
perturbed_ric = propagate_relative_elements_to_ric(
    initial_elements,
    times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=force_model,
    initial_epoch=INITIAL_EPOCH,
    max_step_s=30.0,
    ephemeris_step_s=300.0,
)

print("Final two-body D'Amico ROEs:")
print(two_body_elements[-1, 0:6])
print("Final J2 + Sun osculating D'Amico ROEs:")
print(perturbed_elements[-1, 0:6])

save_relative_trajectory_html(
    perturbed_ric,
    "traj_perturbed_relative_elements.html",
    title="RIC history from perturbed D'Amico initial elements",
)
save_trajectory_diagnostics_html(
    perturbed_ric,
    "diagnostics_perturbed_relative_elements.html",
    frame_kind="relative",
    title="Perturbed relative-element propagation",
)
