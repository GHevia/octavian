# Spherical-harmonic gravity

`SphericalHarmonics` adds zonal, tesseral, and sectoral gravity terms to the
existing point-mass EOM. It works with coast, mass-carrying coast, finite-burn,
low-thrust, and coupled ECI chief/deputy phases, as well as numerical
`propagate.inertial` and `propagate.relative` histories.

```python
import numpy as np
from octavian import Dynamics, Perturbations, SphericalHarmonics

# Illustrative 4x4 coefficients, not a complete terrestrial gravity model.
C = np.zeros((5, 5))
S = np.zeros_like(C)
C[2, 0] = -1.08262668e-3 / np.sqrt(5)  # J2 in fully normalized form
C[2, 2] = 1.5e-6
S[2, 2] = -0.9e-6
C[3, 0] = 0.9e-6
C[4, 0] = 0.5e-6

gravity = SphericalHarmonics(
    cosine=C,
    sine=S,
    reference_radius_m=6_378_136.3,
    degree=4,
    order=4,
    rotation_rate_radps=7.292115e-5,
    reference_angle_rad=0.0,
    reference_time_s=0.0,
    backend="cpp",  # default; "python" remains available explicitly
)
dynamics = Dynamics(perturbations=Perturbations(spherical_harmonics=gravity))
```

Use `dynamics` on an ordinary `Phase`. Quick missions automatically select the
composable compiler when this perturbation is present. To evaluate only the
perturbing acceleration numerically:

```python
acceleration_mps2 = gravity.acceleration(
    [7_000_000, 0, 1_000_000], mu_m3ps2=3.986004418e14, time_s=1200,
)
```

The examples are plain Python scripts: edit the settings near the top and run
without command-line arguments.

```bash
python examples/analysis/02_spherical_harmonics.py
python examples/analysis/03_spherical_harmonics_runtime.py
```

Install `octavian[viz]` for plotting. The propagation example uses an explicit
4×4 coefficient table, with J2 plus illustrative C22/S22, C30, and C40 entries.
It contains no randomness and is not a calibrated Earth gravity model. The
`backend`, `orbits`, and `output` variables control the run. Its PNG compares
point-mass, J2, and spherical-harmonic trajectories; the CSV records the extra
position change beyond J2. RK4 checks propagation consistency using the same
force model; it is not an independent validation of the gravity equations.

The runtime example compares the same dense, deterministic synthetic field and
state through both backends. It reports construction, acceleration, Jacobian,
and adjoint-Hessian times separately, and checks numerical agreement first.
Both paths evaluate ASSET vector functions: the speedup comes from compact
compiled recurrence loops versus repeated ASSET expression subtrees, not simply
from comparing interpreted Python with compiled ASSET. Edit `degrees`, `backends`,
`repeats`, and `trials` in the script. For large fields set `backends = ["cpp"]`.

## Coefficients, units, and orientation

Supply square C/S arrays indexed by `[degree, order]`, with zero entries above
the diagonal and zero `S[:, 0]`. The model copies them into immutable tuples.
Arrays must contain at least degree two. Degree and order can truncate a larger
field without modifying its input arrays.

Coefficients use **fully normalized geodetic associated Legendre functions**,
without the Condon–Shortley sign:

\[
\bar P_{nm}(u)=\sqrt{(2-\delta_{m0})(2n+1)\frac{(n-m)!}{(n+m)!}}P_{nm}(u).
\]

The positive perturbing potential is

\[
U_p=\frac{\mu}{r}\sum_{n=2}^N(R/r)^n
\sum_{m=0}^{\min(n,M)}\bar P_{nm}(\sin\phi)
[\bar C_{nm}\cos(m\lambda)+\bar S_{nm}\sin(m\lambda)],
\quad \mathbf a_p=\nabla U_p.
\]

Convert unnormalized coefficients by dividing them by the normalization factor
above. Schmidt-normalized coefficients need a different conversion; do not
pass them directly. Octavian does not download coefficient files or infer their
normalization. Use the field's published reference radius and gravitational
parameter together. The coefficient radius need not equal the body's mean
radius, which continues to define altitude for other perturbations.

Point-mass gravity remains in the EOM. C00 may be zero or one and is ignored;
degree one must be zero, assuming a center-of-mass origin. `j2=True` together
with a harmonic field is rejected to prevent double counting. A zonal-only
model uses `order=0`; J2 alone has `C[2,0] = -J2/sqrt(5)`.

The coefficient frame rotates about the inertial +Z axis by
`reference_angle_rad + rotation_rate_radps * (time_s - reference_time_s)`.
The angle measures the body's prime meridian from inertial +X toward +Y.
Times are **mission-relative seconds**, including accumulated time across
phases. Choose the angle at your mission epoch explicitly; setting
`initial_epoch` does not set this angle. A zero rotation rate fixes the
coefficient frame. The model does not include precession, nutation, polar
motion, arbitrary pole orientation, or time-varying/tidal coefficients.
The exterior gravity expansion is intended for positions outside the field's
bounding sphere; high-degree expansions can diverge inside it.

## Python and C++ backends

The Python implementation constructs ordinary ASSET vector-function expressions,
consistent with the other EOMs. It also evaluates directly with Python/NumPy for
analysis propagation. A normalized Cartesian solid-harmonic recurrence avoids
longitude and divisions by distance to the pole. Explicit Cartesian derivative identities obtain acceleration from
the next degree of harmonics. ASSET differentiates the resulting
expressions for optimization.

The optional C++ backend runs the recurrence in compiled loops and returns a
native ASSET `VectorFunction`. It evaluates acceleration and its derivatives
without crossing into Python during integration or optimization. Truncated
Taylor arithmetic through third order differentiates the potential analytically;
no finite-difference derivatives or Python callback wrappers are used. Body
rotation is composed outside that function, so ASSET includes the time derivatives.
All evaluation scratch data is local to each call.

The compiled recurrence takes O(NM) work (O(N²) for a square field), and uses
constant recurrence scratch storage in addition to coefficient arrays. The
Python expression tree grows much faster as shared subexpressions are repeated.
Use the Python backend for small fields and transparent inspection; use C++ for
higher degree/order fields. Selecting `cpp` without an installed extension raises
an actionable error and never silently switches backends.

## Pip installation and backend selection

The release build bundles the compiled backend into Octavian wheels for:

| Platform | Python | Native requirements |
| --- | --- | --- |
| Linux x86-64 | 3.10, 3.11, 3.12 | glibc 2.34 or newer, AVX2/FMA |
| Windows x64 | 3.10, 3.11, 3.12 | AVX2, ASSET's MSVC runtime |

On those platforms, `pip install octavian` installs the binary alongside the
Python implementation: no compiler, headers, or ASSET source build is required.
The default is `backend="cpp"`. Select `backend="python"` explicitly for readable
ASSET expressions or installations without the extension. There is no silent
fallback on a missing or incompatible binary. These wheels are delivered when this change is
released; a checkout alone does not install them.

A pure Python wheel and source distribution remain available. Those installations
support the Python backend wherever pip-installed ASSET is supported. They do
not compile native code automatically. A missing or incompatible native binary
raises an error when `cpp` is selected.

## Build the optional standalone extension

Developers can still install the separate local package
`native/spherical_harmonics`, which builds `octavian-harmonics-native`. The loader
prefers the bundled extension when present and otherwise uses this standalone
package. Both use upstream
ASSET headers and the **existing pip-installed `asset_asrl==0.5.1`** at runtime.
It does not build ASSET, modify the wheel, or require a fork. Headers and their
pinned Eigen, pybind11, and fmt submodules are needed only at build time.

On Linux x86-64, in the project environment:

```bash
# Create octavian-dev from environment.yml first, then install Octavian.
conda run -n octavian-dev python -m pip install --no-user -e ".[dev]"
conda install -n octavian-dev -c conda-forge clangxx=18 gcc_impl_linux-64=13 gxx_impl_linux-64=13 cmake ninja
conda run -n octavian-dev python -m pip install scikit-build-core

git clone --branch v0.5.1 --depth 1 https://github.com/AlabamaASRL/asset_asrl.git /tmp/asset-headers
git -C /tmp/asset-headers submodule update --init --depth 1 dep/eigen dep/pybind11 dep/fmt
ASSET_SOURCE_DIR=/tmp/asset-headers conda run -n octavian-dev env CXX=clang++ \
  python -m pip install --no-user --no-build-isolation ./native/spherical_harmonics
conda run -n octavian-dev python -m pytest tests/test_spherical_harmonics.py -q
```

The pinned ASSET tag resolves to `6cb73ac174b140ecc6ac1b9563012424f2f0a748`.
ASSET's native ABI is not a stable public plugin ABI: use its matching headers,
submodules, Clang/libstdc++ ABI, Python minor version, and AVX2/Eigen alignment.
The extension checks for the registered ASSET function type on import and rejects
an incompatible pybind11 ABI. Rebuild and revalidate after changing ASSET versions.
The current native build targets the ASSET x86-64 AVX2 wheel configuration.
Windows uses Clang-CL with MSVC compatibility version 19.40 (matching ASSET's
published pybind11 ABI) and the matching MSVC runtime. The wheel workflow builds
and tests all six platform/Python combinations against pip-installed ASSET,
including derivatives, optimization, the propagation example, and runtime comparison. Publishing
waits for every wheel test to pass.

Maintainers can build a bundled wheel with `OCTAVIAN_BUILD_NATIVE=1 python -m build
--wheel`, with `ASSET_SOURCE_DIR`, Clang, CMake, and Ninja configured as above.
The build hook is opt-in; regular editable/source installations remain compiler
free. Linux release wheels are repaired with auditwheel before installation tests.

The Python backend needs no compiler or header checkout. CWH, CR3BP, and relative
formulations other than coupled ECI reject harmonic perturbations, consistently
with their existing force-model restrictions.

## Accuracy checks and degree limits

`tests/test_spherical_harmonics.py` checks:

- C20 against the existing analytic J2 acceleration, including the poles.
- Acceleration against finite differences of an independent latitude/longitude
  Legendre potential through 20×20, including a truncated-order field.
- A closed-form single Cnn/Snn term through degree 100, using factorial
  normalization rather than the implementation's recurrence.
- Python/C++ agreement, pole behavior, Jacobian and adjoint-Hessian finite
  differences at 50×50 and 100×100; symmetry and zero divergence of the exterior
  gravity gradient for high-degree sectoral fields.
- Body rotation/time derivatives, integration, and solver optimization.

These are numerical implementation checks, not certification of a measured
Earth model or arbitrary degrees. For mission validation, compare a published
coefficient model against an independent trusted gravity implementation using
identical coefficients, normalization, GM, radius, body orientation, and epoch.
Also check degree-truncation and integration-tolerance convergence. Current
orientation is uniform rotation about Z, not a full Earth orientation model.

There is no hard-coded maximum degree/order. Arrays of shape `(N+1, N+1)` support
maximum degree N, with order M ≤ N. Current regression coverage reaches 100×100;
higher degrees need their own accuracy and performance validation. For full
order, degrees 2…N contain `(N+1)**2 - 4` potentially nonzero real coefficients
(C plus S): 437 at 20×20, 2,597 at 50×50, and 10,197 at 100×100. The arrays also
contain required zero/unused slots. Zero-padding a low-degree field does not
add physical detail. Work scales approximately quadratically for the native
square-field recurrence; the Python ASSET expression tree becomes expensive
around 10×10.

## What the cosine and sine inputs mean

`cosine[n, m]` is the dimensionless fully normalized coefficient Cbar_nm;
`sine[n, m]` is Sbar_nm. They are amplitudes, not angles or precomputed trig
functions. They multiply `cos(m * longitude)` and `sin(m * longitude)` in the
potential above. Degree n describes spatial complexity; order m describes the
longitude dependence. For m=0 the term is longitude-independent (zonal), and
S[n,0] is zero. C22/S22 together specify a degree-two longitude pattern.

Use coefficients from a gravity model for the body, with its published GM and
reference radius. The example's small explicit table demonstrates the API;
the benchmark's deterministic values exercise every term without requiring a
data download. Neither is a measured gravity model. See the
[ICGEM coefficient explanation](https://icgem.gfz.de/faq) and
[model file format](https://icgem.gfz.de/docs/ICGEM-Format-2023.pdf).
