"""Relative example 25: differential cannonball drag and SRP.

Chief and deputy use different constant projected areas.  The analysis API
converts D'Amico elements to absolute states, propagates both spacecraft with
J2, exponential drag, and BSP-driven SRP, then reconstructs osculating
relative elements and RIC histories.
"""

from __future__ import annotations

import numpy as np

from octavian import (
    EARTH,
    Cannonball,
    Dynamics,
    Perturbations,
    Spacecraft,
    state,
)
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
CHIEF_SEMI_MAJOR_AXIS_M = EARTH.mean_radius_m + 500_000.0

chief_position, chief_velocity = classical_to_cartesian(
    a_m=CHIEF_SEMI_MAJOR_AXIS_M,
    e=0.001,
    inc_deg=51.6,
    raan_deg=20.0,
    argp_deg=10.0,
    true_anomaly_deg=30.0,
    mu_m3ps2=EARTH.mu_m3ps2,
)
chief_eci = state(chief_position, chief_velocity)
initial_elements = RelativeOrbitalElements(
    delta_a=100.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_lambda_rad=-2_000.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ex=300.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ey=-100.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_ix_rad=200.0 / CHIEF_SEMI_MAJOR_AXIS_M,
    delta_iy_rad=-150.0 / CHIEF_SEMI_MAJOR_AXIS_M,
)

chief_spacecraft = Spacecraft(
    name="Chief",
    dry_mass_kg=300.0,
    cannonball=Cannonball(
        drag_area_m2=2.0,
        drag_coefficient=2.2,
        srp_area_m2=3.0,
        reflectivity_coefficient=1.3,
    ),
)
deputy_spacecraft = Spacecraft(
    name="Deputy",
    dry_mass_kg=150.0,
    cannonball=Cannonball(
        drag_area_m2=4.0,
        drag_coefficient=2.2,
        srp_area_m2=5.0,
        reflectivity_coefficient=1.4,
    ),
)
force_model = Perturbations(j2=True, drag=True, srp=True)

# This is the same dynamics declaration used by composable coast and finite-
# burn phases. The phase Spacecraft is the deputy; chief properties live on
# the relative model.
relative_dynamics = Dynamics.relative(
    chief_initial_state_eci=chief_eci,
    chief_spacecraft=chief_spacecraft,
    propagation_mode="coupled_eci",
    perturbations=force_model,
)
assert relative_dynamics.active_perturbations().drag

times_s = np.linspace(0.0, 2.0 * 3_600.0, 97)
element_history = propagate_relative_orbital_elements(
    initial_elements,
    times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=force_model,
    initial_epoch=INITIAL_EPOCH,
    chief_spacecraft=chief_spacecraft,
    deputy_spacecraft=deputy_spacecraft,
    max_step_s=20.0,
    ephemeris_step_s=300.0,
)
ric_history = propagate_relative_elements_to_ric(
    initial_elements,
    times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=force_model,
    initial_epoch=INITIAL_EPOCH,
    chief_spacecraft=chief_spacecraft,
    deputy_spacecraft=deputy_spacecraft,
    max_step_s=20.0,
    ephemeris_step_s=300.0,
)

print("Initial D'Amico ROEs:")
print(element_history[0, 0:6])
print("Final osculating D'Amico ROEs with J2 + drag + SRP:")
print(element_history[-1, 0:6])

save_relative_trajectory_html(
    ric_history,
    "traj_cannonball_drag_srp.html",
    title="Differential cannonball drag and SRP",
)
save_trajectory_diagnostics_html(
    ric_history,
    "diagnostics_cannonball_drag_srp.html",
    frame_kind="relative",
    title="Differential cannonball force diagnostics",
)
