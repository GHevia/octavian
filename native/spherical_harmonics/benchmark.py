"""Compare ASSET harmonic force/derivative evaluation for the same field.

Run: conda run -n octavian-dev python native/spherical_harmonics/benchmark.py
Times exclude construction, imports, and a warm-up. They include the Python
call into ASSET on both sides. Values are synthetic, fully normalized fields.
"""

import argparse
import timeit
from dataclasses import replace
from functools import partial

import numpy as np

from octavian import SphericalHarmonics
from octavian._asset import vf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--degrees", type=int, nargs="+", default=[4, 8, 10])
parser.add_argument("--repeats", type=int, default=10)
parser.add_argument("--native-only", action="store_true")
options = parser.parse_args()
if options.repeats < 1 or any(n < 2 for n in options.degrees):
    parser.error("repeats must be positive and degrees must be at least 2")
print("degree,backend,build_s,force_us,jacobian_us,hessian_us")
for degree in options.degrees:
    rng = np.random.default_rng(42)
    cosine = np.tril(rng.normal(0, 1e-6, (degree + 1, degree + 1)))
    sine = np.tril(rng.normal(0, 1e-6, (degree + 1, degree + 1)))
    cosine[:2] = 0
    sine[:2] = 0
    sine[:, 0] = 0
    cosine[2, 0] = -1.08262668e-3 / np.sqrt(5)
    field = SphericalHarmonics(cosine, sine, 6378136.3)
    reference = None
    for backend in (["cpp"] if options.native_only else ["python", "cpp"]):
        gravity = replace(field, backend=backend)
        args = vf.Arguments(3)
        start = timeit.default_timer()
        function = gravity.asset_acceleration(
            args * gravity.reference_radius_m, 0.0, mu_m3ps2=3.986004418e14
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
            timings.append(
                min(timeit.repeat(call, number=options.repeats, repeat=3)) / options.repeats * 1e6
            )
        print(
            f"{degree},{backend},{build_s:.6g}," + ",".join(f"{value:.6g}" for value in timings),
            flush=True,
        )
