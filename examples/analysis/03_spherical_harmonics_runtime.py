"""Compare ASSET harmonic force/derivative evaluation for the same field.

Run this file normally; edit the settings below.
Times exclude construction, imports, and a warm-up. They include the Python
call into ASSET on both sides. Values are the published fully normalized EGM2008 coefficients.
"""

import timeit
from dataclasses import replace
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

print("degree,backend,build_s,force_us,jacobian_us,hessian_us")
for degree in degrees:
    field = SphericalHarmonics.earth(degree=degree, rotation_rate_radps=0.0)
    reference = None
    measurements = {}
    for backend in backends:
        gravity = replace(field, backend=backend)
        args = vf.Arguments(3)
        start = timeit.default_timer()
        function = gravity.asset_acceleration(
            args * gravity.reference_radius_m, 0.0, mu_m3ps2=field.reference_mu_m3ps2
        )
        build_s = timeit.default_timer() - start
        point = np.array([1.1, 0.2, -0.3])
        adjoint = np.array([0.2, 0.3, 0.7])
        actual = (
            function.compute(point),
            function.jacobian(point),
            function.adjointhessian(point, adjoint),
        )
        if reference is not None:
            for result, expected in zip(actual, reference, strict=True):
                np.testing.assert_allclose(result, expected, rtol=2e-11, atol=1e-13)
        reference = actual
        timings = []
        for call in [
            partial(function.compute, point),
            partial(function.jacobian, point),
            partial(function.adjointhessian, point, adjoint),
        ]:
            timings.append(min(timeit.repeat(call, number=repeats, repeat=trials)) / repeats * 1e6)
        print(
            f"{degree},{backend},{build_s:.6g}," + ",".join(f"{value:.6g}" for value in timings),
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
