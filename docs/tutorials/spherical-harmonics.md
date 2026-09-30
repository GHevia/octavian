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
    backend="python",  # or "cpp" to select the bundled compiled backend
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

Run the same propagation with either backend:

```bash
python examples/analysis/02_spherical_harmonics.py --backend python --output python-gravity.png
python examples/analysis/02_spherical_harmonics.py --backend cpp --degree 20 --output cpp-gravity.png
```

Install `octavian[viz]` for plotting. Each run propagates point-mass, J2, and
spherical-harmonic gravity from an identical initial state for two orbits. The
PNG shows the orbit paths, departure from point-mass gravity, and the additional
position change beyond J2. The adjacent CSV contains that additional change in
meters. The example also checks ASSET propagation against numerical RK4 over
600 seconds. Higher-order coefficients are synthetic, not a calibrated Earth
model. The default 4×4 case produces about 108 m of additional displacement
beyond J2; use the same degree to compare the two backends.

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
Use `backend="python"` (the default) for readable ASSET expressions, or explicitly
select `backend="cpp"` for compiled evaluation. Installing the binary does not
change the selected backend. These wheels are delivered when this change is
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
including derivatives, optimization, and the 20×20 propagation example. Publishing
waits for every wheel test to pass.

Maintainers can build a bundled wheel with `OCTAVIAN_BUILD_NATIVE=1 python -m build
--wheel`, with `ASSET_SOURCE_DIR`, Clang, CMake, and Ninja configured as above.
The build hook is opt-in; regular editable/source installations remain compiler
free. Linux release wheels are repaired with auditwheel before installation tests.

The Python backend needs no compiler or header checkout. CWH, CR3BP, and relative
formulations other than coupled ECI reject harmonic perturbations, consistently
with their existing force-model restrictions.

## Reproduce timings

Run `conda run -n octavian-dev python native/spherical_harmonics/benchmark.py`
from the checkout for matching-field ASSET force, Jacobian, and adjoint-Hessian
timings at 4x4, 8x8, and 10x10. Construction is reported separately; calls are
warmed up and checked for numerical agreement. For larger native-only cases,
add `--native-only --degrees 20 50 100`. Synthetic coefficients are used so the
benchmark has no gravity-data download dependency.
