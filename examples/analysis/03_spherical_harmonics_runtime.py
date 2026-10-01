"""Compare ASSET harmonic force/derivative evaluation for the same field.

Run this file normally; edit the settings below.
Times exclude construction, imports, and a warm-up. They include the Python
call into ASSET on both sides. Values are the published fully normalized EGM2008 coefficients.
"""

import timeit
from functools import partial

import numpy as np

from octavian import SphericalHarmonics
from octavian._asset import vf

# Compare identical fields and states. Python expressions become expensive
# around 10x10; choose backends=["cpp"] for larger degrees such as 20, 50, 100.
degrees = [4, 8, 10]
backends = ["python", "cpp"]
repeats = 3
trials = 3

print("Construction time is reported separately from repeated evaluations.")
print("Jacobians and Hessians are the derivatives ASSET uses during optimization.")
print(
    f"{'Field':>8} {'Backend':>8} {'Build [s]':>12} {'Force [µs]':>12} {'Jacobian [µs]':>15} {'Hessian [µs]':>15}"
)
for degree in degrees:
    measurements = {}
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
    if "python" in measurements and "cpp" in measurements:
        speedups = np.asarray(measurements["python"]) / measurements["cpp"]
        print(
            f"{degree}x{degree} speedup (force/Jacobian/Hessian): "
            + " / ".join(f"{speedup:.1f}x" for speedup in speedups),
            flush=True,
        )
