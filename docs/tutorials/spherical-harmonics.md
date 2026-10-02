# Spherical-harmonic gravity

`SphericalHarmonics` adds zonal, tesseral, and sectoral gravity terms to the
existing point-mass EOM. It works with coast, mass-carrying coast, finite-burn,
low-thrust, and coupled ECI chief/deputy phases, as well as numerical
`propagate.inertial` and `propagate.relative` histories.

## Published Earth coefficients: no manual arrays required

```python
from octavian import Dynamics, Perturbations, SphericalHarmonics

gravity = SphericalHarmonics.earth()  # EGM2008, degree/order 200, C++ backend
dynamics = Dynamics(
    perturbations=Perturbations(spherical_harmonics=gravity),
)
```

This loads a **bundled, offline 200×200 subset of NGA EGM2008**. No coefficient
calculation, network access, Java, or database account is required. NGA recommends
[EGM2008 over legacy EGM96](https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84).
The full published model extends beyond this subset; Octavian bundles through
200×200 and rejects larger requests rather than silently padding with zeros.

```python
gravity = SphericalHarmonics.earth(degree=20, order=20)
small_python_field = SphericalHarmonics.earth(degree=4, backend="python")
```

The factory loads fully normalized, **tide-free** C/S values, the published
reference radius (6378136.3 m), and reference GM (3.986004415e14 m³/s²).
The selected field supplies GM automatically to `Dynamics`, ASSET EOMs and
numerical propagation. It takes precedence over the general central-body GM,
so point-mass and harmonic acceleration always use the same published constant.
No replacement of `EARTH`, coefficient conversion, or manual parameter copying
is needed. Custom fields without `reference_mu_m3ps2` use the caller's GM.

```python
from octavian import propagate, state
from octavian.dynamics import PerturbedECI

forces = Perturbations(spherical_harmonics=SphericalHarmonics.earth(degree=20, order=20))
history = propagate.inertial(state([7e6, 0, 0], [0, 7500, 0]), [0, 600], perturbations=forces)
ode = PerturbedECI(spherical_harmonics=forces.spherical_harmonics)
```

Uniform Z rotation defaults to 7.292115e-5 rad/s. Set `reference_angle_rad` and
`reference_time_s` to your mission's orientation convention; the factory does
not turn an arbitrary mission epoch into an ITRF transform. Tides, nutation,
precession, polar motion, and time-varying coefficients remain separate work.

The dataset manifest records source URL, original archive SHA-256, uncompressed
subset SHA-256, constants, tide system, and citation. The runtime checks the
subset checksum. `tools/prepare_egm2008.py` reproduces it from NGA's archive;
coefficient digits are retained unchanged. The gzip subset adds about 636 KiB.
Source: Pavlis et al. (2012), [EGM2008](https://doi.org/10.1029/2011JB008916).
The [ICGEM model database](https://icgem.gfz.de/tom_longtime) provides other
published models; automatic network fetching and a generic model-file importer
are not part of this API. Custom arrays still work through the constructor.

The examples are plain Python scripts with settings near the top:

```bash
python examples/analysis/02_spherical_harmonics.py
python examples/analysis/03_spherical_harmonics_runtime.py
```

Install `octavian[viz]` for plotting. Example 02 compares point-mass, model-matched
J2, and measured EGM2008 gravity, saving PNG/CSV outputs. Edit `degree`, `backend`,
`orbits`, and `output`; use a small degree such as 4 for Python ASSET expressions.
Independent validation lives in Actium; example 02 focuses on the gravity model’s
effect on an orbit. Example 03
compares both backends with identical EGM2008 coefficients at 4×4, 8×8, and
10×10, including construction, force, Jacobian, and adjoint-Hessian timings.

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
pass them directly. The Earth factory supplies this convention automatically; custom arrays must
use it explicitly. Octavian does not infer custom-array normalization. Use the field's published reference radius and gravitational
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

Source distributions and editable installations compile the same extension by
default. There is no separate native-package installation step.

## Editable and source installations

Install Git and a compatible C++ toolchain once:

- **Linux x86-64:** Clang 17/18 and C++ standard-library development headers
  (the validated Conda toolchain is `clangxx=18 gcc_impl_linux-64=13 gxx_impl_linux-64=13`).
- **Windows x64:** Visual Studio 2022 C++ Build Tools and LLVM/Clang-CL.
  Run installation in an x64 Visual Studio developer terminal. The build hook
  sets MSVC compatibility version 19.40 to match pip ASSET 0.5.1.
- **Python:** the supported native matrix is 3.10–3.12. A system Python also
  needs its development headers; Conda supplies these with Python.

With the compiler available on `PATH`, a venv uses the normal installation command:

```bash
python -m pip install -e ".[dev]"
python -c "from octavian import SphericalHarmonics; print(SphericalHarmonics.earth(degree=200).acceleration([7e6, 1e6, 2e6]))"
```

Pip supplies Hatchling, CMake, and Ninja in its isolated build environment. The
build downloads ASSET revision `6cb73ac174b140ecc6ac1b9563012424f2f0a748`
and its pinned Eigen, pybind11, and fmt submodules into `.native-build/`.
Only the gravity extension is compiled; the runtime remains the existing
`asset_asrl==0.5.1` wheel. Initial source builds require network access to GitHub.
To use an existing checkout or build offline, set `ASSET_SOURCE_DIR` to the pinned
ASSET source with its submodules initialized. Set `CXX` to select a compiler explicitly.

Editable builds place the ABI-tagged extension inside the checkout's `octavian/`
directory. Python edits take effect immediately. After changing C++ sources,
rerun `python -m pip install -e .`; close processes using the extension first,
particularly on Windows. Generated binaries, license copies, and cached headers
are ignored by Git. Use separate checkouts for environments requiring different
native ABIs; pip uninstall does not remove an in-place editable binary.

A compiler-free developer installation is an explicit exception:

```bash
OCTAVIAN_BUILD_NATIVE=0 python -m pip install -e .
```

That configuration requires `backend="python"` for harmonic evaluation. It does
not remove an existing in-place binary. Native build failures are errors; the
installer never silently substitutes a Python-only build. The separately built
legacy extension remains import-compatible, but is no longer needed for normal
installation.

Release wheels are built by the same default hook and tested outside the source
checkout. The six Linux/Windows and Python 3.10–3.12 jobs also test default editable
installs in fresh environments. Publishing ships those native wheels plus the
source distribution, without a competing pure-Python wheel.

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
maximum degree N, with order M ≤ N. Synthetic regression coverage reaches 100×100; the separate Actium/Orekit
campaign validates measured EGM2008 fields through 200×200. Higher degrees need
their own accuracy and performance validation. For full
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
reference radius. The Earth factory and both examples use measured EGM2008 coefficients; custom
arrays remain available for other bodies or test fields. See the
[ICGEM coefficient explanation](https://icgem.gfz.de/faq) and
[model file format](https://icgem.gfz.de/docs/ICGEM-Format-2023.pdf).

## Independent validation through Actium/Orekit

The independent [Actium repository](https://github.com/GHevia/actium) provides
[`validation/validate_spherical_harmonics.py`](https://github.com/GHevia/actium/blob/main/validation/validate_spherical_harmonics.py).
Its installable reference package never imports Octavian. Orekit independently
parses the same ICGEM coefficient file and evaluates Holmes–Featherstone gravity;
the bridge then compares Octavian/ASSET and Orekit numerical propagation.

Run in the Octavian development environment after installing the Actium checkout
and Java 11+ (or Actium's `jdk` extra):

```bash
git clone https://github.com/GHevia/actium.git ../actium
PYTHONNOUSERSITE=1 conda run -n octavian-dev python -m pip install -e "../actium[dev,jdk]"
PYTHONNOUSERSITE=1 conda run -n octavian-dev python ../actium/validation/validate_spherical_harmonics.py
```

The introductory script compares one 200×200 orbit in five sequential steps.
The detailed campaign in `validation/regression/spherical_harmonics.py` chooses
20×20, 100×100, and 200×200, six hours and 121
identical samples for inclined and near-polar LEO, with two tolerance settings.
It verifies every coefficient and both constants via independent readers,
evaluates forces at identical states (including near-pole and high-altitude
points), and compares position/velocity/acceleration histories. Tolerance
sensitivity is recorded for each integrator. CSV, PNG, and JSON reports are
written under Actium's `validation/results/spherical_harmonics` before any
nonzero failure exit. Gates are 1 mm, 1 µm/s, 1e-9 m/s² history acceleration,
and 1e-11 m/s² same-state force disagreement.

Both engines use identical uniform Z rotation and point mass plus static
EGM2008 only. This validates the implemented gravity/EOM and integration; it
does not claim equivalence to a full high-fidelity ITRF/tides/drag/SRP model.
See [Actium’s validation guide](https://github.com/GHevia/actium/blob/main/validation/README.md)
and the generated `summary.json` for measured results.

### Recorded six-hour EGM2008 result

All 12 comparisons passed (two orbit types × three degrees × two tolerance
settings). Each row reports the maximum across both orbit cases and both
settings, over all 121 samples:

| degree/order | position [m] | velocity [m/s] | history acceleration [m/s²] | same-state force [m/s²] |
| --- | ---: | ---: | ---: | ---: |
| 20×20 | 2.168e-04 | 1.362e-07 | 4.956e-10 | 7.692e-17 |
| 100×100 | 2.085e-04 | 3.112e-07 | 2.805e-10 | 2.187e-16 |
| 200×200 | 3.267e-04 | 4.007e-07 | 4.284e-10 | 4.165e-16 |

These bounds apply to the documented matched-physics campaign, not arbitrary
missions or full Earth-orientation models. The recorded reference uses
Orekit-JPype 13.1.8.0. See Actium’s `validation/results/spherical_harmonics/summary.json` for every case and convergence metrics.
