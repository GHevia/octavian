"""Compare ASSET harmonic evaluations and ODE integration for the same field.

Run this file normally; edit the settings below.
Times exclude construction, imports, and a warm-up. They include the Python
call into ASSET on both sides. Values are the published fully normalized EGM2008 coefficients.
"""

import timeit
from functools import partial

import numpy as np

from octavian import SphericalHarmonics
from octavian._asset import vf
from octavian.dynamics import PerturbedECI

# Compare identical fields and states. Python expressions become expensive
# around 10x10; choose backends=["cpp"] for larger degrees such as 20, 50, 100.
degrees = [4, 8, 10]
backends = ["python", "cpp"]
repeats = 3
trials = 3
integration_trials = 1  # Full Python ODE propagations are much slower than individual calls.
orbits = 1.0
initial_step_s = 10.0
absolute_tolerance = 1e-7  # SI states, as in example 02.
output_points = 401
initial_row = np.array([7e6, 0.0, 2e5, 0.0, 7400.0, 800.0, 0.0])

print("Construction time is reported separately from repeated evaluations.")
print("Jacobians and Hessians are the derivatives ASSET uses during optimization.")
print(
    f"{'Field':>8} {'Backend':>8} {'Build [s]':>12} {'Force [µs]':>12} {'Jacobian [µs]':>15} {'Hessian [µs]':>15}"
)
for degree in degrees:
    measurements = {}
    integration_times = {}
    histories = {}
    for backend in backends:
        gravity = SphericalHarmonics.earth(degree=degree, backend=backend, rotation_rate_radps=0.0)
        # The three ASSET inputs are position components in reference-radius units.
        args = vf.Arguments(3)
        start = timeit.default_timer()
        function = gravity.asset_acceleration(args * gravity.reference_radius_m, 0.0)
        build_s = timeit.default_timer() - start
        point = np.array([1.1, 0.2, -0.3])
        # Fixed weights combine the three acceleration Hessians into one matrix.
        adjoint = np.array([0.2, 0.3, 0.7])
        # Warm up each operation before measuring repeated evaluations.
        function.compute(point)
        function.jacobian(point)
        function.adjointhessian(point, adjoint)
        timings = []
        for call in [
            partial(function.compute, point),
            partial(function.jacobian, point),
            partial(function.adjointhessian, point, adjoint),
        ]:
            timings.append(min(timeit.repeat(call, number=repeats, repeat=trials)) / repeats * 1e6)
        print(
            f"{degree:>3}×{degree:<4} {backend:>8} {build_s:12.4f} "
            f"{timings[0]:12.3f} {timings[1]:15.3f} {timings[2]:15.3f}",
            flush=True,
        )
        measurements[backend] = timings

        # Propagate the full six-state ODE with the same static field used above.
        # ODE/integrator construction and warm-up stay outside the timed calls.
        start = timeit.default_timer()
        ode = PerturbedECI(spherical_harmonics=gravity)
        integrator = ode.integrator(initial_step_s)
        integrator.setAbsTol(absolute_tolerance)
        ode_build_s = timeit.default_timer() - start
        mu = gravity.reference_mu_m3ps2
        energy = np.dot(initial_row[3:6], initial_row[3:6]) / 2 - mu / np.linalg.norm(initial_row[:3])
        semi_major_axis_m = -mu / (2 * energy)
        duration_s = orbits * 2 * np.pi * np.sqrt(semi_major_axis_m**3 / mu)
        propagate = partial(integrator.integrate_dense, initial_row, duration_s, output_points)
        history = np.asarray(propagate())
        if history.shape != (output_points, 7) or not np.isfinite(history).all():
            raise RuntimeError(f"Invalid integration history for {degree}x{degree} {backend}")
        np.testing.assert_allclose(history[[0, -1], 6], [0.0, duration_s], atol=1e-9, rtol=0)
        histories[backend] = history
        integration_s = min(timeit.repeat(propagate, number=1, repeat=integration_trials))
        integration_times[backend] = integration_s
        print(
            f"  {backend}: ODE/integrator build {ode_build_s:.4f} s; "
            f"integrate {orbits:g} orbit(s), {output_points} samples: {integration_s:.6f} s",
            flush=True,
        )
    if "python" in measurements and "cpp" in measurements:
        speedups = np.asarray(measurements["python"]) / measurements["cpp"]
        print(
            f"{degree}x{degree} speedup (force/Jacobian/Hessian): "
            + " / ".join(f"{speedup:.1f}x" for speedup in speedups),
            flush=True,
        )
        position_difference_m = np.linalg.norm(histories["python"][:, :3] - histories["cpp"][:, :3], axis=1)
        velocity_difference_mps = np.linalg.norm(histories["python"][:, 3:6] - histories["cpp"][:, 3:6], axis=1)
        np.testing.assert_allclose(histories["python"][:, 6], histories["cpp"][:, 6], atol=1e-9, rtol=0)
        np.testing.assert_allclose(histories["python"][:, :6], histories["cpp"][:, :6], atol=1e-4, rtol=1e-10)
        print(
            f"{degree}x{degree} integration speedup: "
            f"{integration_times['python'] / integration_times['cpp']:.1f}x; "
            f"maximum trajectory difference: {position_difference_m.max():.3g} m, "
            f"{velocity_difference_mps.max():.3g} m/s",
            flush=True,
        )
