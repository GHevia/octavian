"""Propagate an illustrative harmonic gravity field with ASSET.

Run with --backend python for readable ASSET expressions, or --backend cpp
for the optional compiled recurrence. --degree 20 demonstrates a larger field.
Coefficients above J2 are synthetic demonstration data, not an Earth model.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from octavian import EARTH, Perturbations, SphericalHarmonics, propagate, state
from octavian.dynamics import PerturbedECI

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("python", "cpp"), default="python")
parser.add_argument("--degree", type=int, default=4)
options = parser.parse_args()
if options.degree < 2:
    parser.error("--degree must be at least 2")

rng = np.random.default_rng(42)
cosine = np.zeros((options.degree + 1, options.degree + 1))
sine = np.zeros_like(cosine)
for n in range(2, options.degree + 1):
    cosine[n, : n + 1] = rng.normal(0, 1e-6 / n**2, n + 1)
    sine[n, 1 : n + 1] = rng.normal(0, 1e-6 / n**2, n)
cosine[2, 0] = -EARTH.j2_coefficient / np.sqrt(5)
gravity = SphericalHarmonics(
    cosine=cosine,
    sine=sine,
    reference_radius_m=EARTH.mean_radius_m,
    rotation_rate_radps=7.292115e-5,
    reference_angle_rad=0.0,  # Prime-meridian angle at mission-relative time zero.
    backend=options.backend,
)
initial = state([7e6, 0, 2e5], [0, 7400, 800])
start = time.perf_counter()
ode = PerturbedECI(mu_m3ps2=EARTH.mu_m3ps2, spherical_harmonics=gravity)
build_seconds = time.perf_counter() - start
integrator = ode.integrator(10.0)
integrator.setAbsTol(1e-7)  # SI states: avoid chasing sub-roundoff position errors.
start = time.perf_counter()
final = np.asarray(integrator.integrate(np.r_[initial.r_m, initial.v_mps, 0.0], 600.0))
integration_seconds = time.perf_counter() - start

# The same configuration also works with analysis propagation and Dynamics on a Phase.
reference = propagate.inertial(
    initial,
    [0, 600],
    perturbations=Perturbations(spherical_harmonics=gravity),
    max_step_s=2,
)
position_error_m = np.linalg.norm(final[:3] - reference[-1, :3])
if position_error_m > 0.05:
    raise RuntimeError(f"ASSET / numerical propagation disagreement: {position_error_m:g} m")
print(f"Backend: {options.backend}; degree/order: {gravity.degree}/{gravity.order}")
print(f"Build: {build_seconds:.3f} s; ASSET propagation: {integration_seconds:.3f} s")
print(f"Final position [m]: {final[:3]}")
print(f"ASSET / RK4 position difference: {position_error_m:.6g} m")
