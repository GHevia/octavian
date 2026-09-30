"""Compare point-mass, J2, and spherical-harmonic gravity with ASSET.

Run with --backend python for ASSET expressions or --backend cpp for the
compiled recurrence. --degree 20 demonstrates a larger field. The PNG shows
orbital paths and accumulated position differences from the same initial state.
Coefficients above J2 are synthetic demonstration data, not an Earth model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from octavian import EARTH, Perturbations, SphericalHarmonics, propagate, state
from octavian.dynamics import PerturbedECI, TwoBodyECI

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("python", "cpp"), default="python")
parser.add_argument("--degree", type=int, default=4)
parser.add_argument("--orbits", type=float, default=2.0)
parser.add_argument("--output", type=Path, default=Path("traj_spherical_harmonics.png"))
options = parser.parse_args()
if options.degree < 2:
    parser.error("--degree must be at least 2")
if not np.isfinite(options.orbits) or options.orbits <= 0:
    parser.error("--orbits must be finite and positive")

# Deterministic illustrative coefficients; C20 reproduces Octavian's existing J2.
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
initial_row = np.r_[initial.r_m, initial.v_mps, 0.0]
energy = np.dot(initial.v_mps, initial.v_mps) / 2 - EARTH.mu_m3ps2 / np.linalg.norm(initial.r_m)
semi_major_axis_m = -EARTH.mu_m3ps2 / (2 * energy)
period_s = 2 * np.pi * np.sqrt(semi_major_axis_m**3 / EARTH.mu_m3ps2)
duration_s = options.orbits * period_s

start = time.perf_counter()
harmonic_ode = PerturbedECI(mu_m3ps2=EARTH.mu_m3ps2, spherical_harmonics=gravity)
build_seconds = time.perf_counter() - start
models = {
    "Point mass": TwoBodyECI(mu_m3ps2=EARTH.mu_m3ps2),
    "J2": PerturbedECI(mu_m3ps2=EARTH.mu_m3ps2, j2=True),
    "Spherical harmonics": harmonic_ode,
}
histories = {}
for name, ode in models.items():
    integrator = ode.integrator(10.0)
    integrator.setAbsTol(1e-7)  # SI states: avoid chasing sub-roundoff position errors.
    start = time.perf_counter()
    histories[name] = np.asarray(integrator.integrate_dense(initial_row, duration_s, 401))
    print(f"{name}: {time.perf_counter() - start:.3f} s for {options.orbits:g} orbits")

# Verify the force model independently with Octavian's Python RK4 propagator.
check_time_s = min(600.0, duration_s)
reference = propagate.inertial(
    initial,
    [0, check_time_s],
    perturbations=Perturbations(spherical_harmonics=gravity),
    max_step_s=2,
)
integrator = harmonic_ode.integrator(10.0)
integrator.setAbsTol(1e-7)
check_state = integrator.integrate(initial_row, check_time_s)
position_error_m = np.linalg.norm(check_state[:3] - reference[-1, :3])
if position_error_m > 0.05:
    raise RuntimeError(f"ASSET / numerical propagation disagreement: {position_error_m:g} m")

point_mass = histories["Point mass"]
j2 = histories["J2"]
harmonics = histories["Spherical harmonics"]
np.testing.assert_allclose(j2[:, 6], point_mass[:, 6], rtol=0, atol=1e-9)
np.testing.assert_allclose(harmonics[:, 6], point_mass[:, 6], rtol=0, atol=1e-9)
minutes = harmonics[:, 6] / 60
extra_position_m = harmonics[:, :3] - j2[:, :3]
extra_distance_m = np.linalg.norm(extra_position_m, axis=1)

figure = plt.figure(figsize=(13, 7), layout="constrained")
grid = figure.add_gridspec(2, 2)
orbit_axes = figure.add_subplot(grid[:, 0], projection="3d")
colors = {"Point mass": "#777777", "J2": "#d48806", "Spherical harmonics": "#1672a5"}
for name, history in histories.items():
    orbit_axes.plot(*history[:, :3].T / 1e3, label=name, color=colors[name], linewidth=1.5)
orbit_axes.scatter(*initial.r_m / 1e3, color="black", s=25, label="Shared initial state")
orbit_axes.set(xlabel="Inertial X [km]", ylabel="Inertial Y [km]", zlabel="Inertial Z [km]")
limit_km = np.max(np.linalg.norm(harmonics[:, :3], axis=1)) / 1e3
orbit_axes.set(xlim=(-limit_km, limit_km), ylim=(-limit_km, limit_km), zlim=(-limit_km, limit_km))
orbit_axes.set_box_aspect((1, 1, 1))
orbit_axes.set_title("Orbit paths nearly overlap at this scale")
orbit_axes.legend(loc="upper left", fontsize=9)

separation_axes = figure.add_subplot(grid[0, 1])
for name in ("J2", "Spherical harmonics"):
    separation_km = np.linalg.norm(histories[name][:, :3] - point_mass[:, :3], axis=1) / 1e3
    separation_axes.plot(minutes, separation_km, label=name, color=colors[name])
separation_axes.set(
    title="Accumulated departure from point-mass gravity", ylabel="Position difference [km]"
)
separation_axes.legend()
separation_axes.grid(alpha=0.25)

extra_axes = figure.add_subplot(grid[1, 1], sharex=separation_axes)
for index, label in enumerate(("X", "Y", "Z")):
    extra_axes.plot(minutes, extra_position_m[:, index], label=label, linewidth=1.2)
extra_axes.plot(minutes, extra_distance_m, color="black", label="Magnitude", linewidth=1.8)
extra_axes.set(
    title="Effect of terms beyond J2: harmonics minus J2",
    xlabel="Elapsed time [min]",
    ylabel="Position difference [m]",
)
extra_axes.legend(ncols=4, fontsize=9)
extra_axes.grid(alpha=0.25)
figure.suptitle(
    f"Rotating {gravity.degree}×{gravity.order} gravity field · {options.backend} backend\nSynthetic coefficients beyond J2; identical initial states",
    fontsize=14,
)
options.output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(options.output, dpi=170)
plt.close(figure)

csv_path = options.output.with_suffix(".csv")
np.savetxt(
    csv_path,
    np.column_stack([harmonics[:, 6], extra_position_m, extra_distance_m]),
    delimiter=",",
    header="time_s,harmonics_minus_j2_x_m,harmonics_minus_j2_y_m,harmonics_minus_j2_z_m,position_difference_m",
    comments="",
)
print(
    f"Backend: {options.backend}; degree/order: {gravity.degree}/{gravity.order}; build: {build_seconds:.3f} s"
)
print(f"Maximum effect beyond J2: {extra_distance_m.max():.6g} m")
print(f"Final effect beyond J2: {extra_distance_m[-1]:.6g} m")
print(f"ASSET / RK4 position difference at {check_time_s:g} s: {position_error_m:.6g} m")
print(f"Wrote: {options.output} and {csv_path}")
