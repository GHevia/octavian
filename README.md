<p align="center">
  <img src="assets/octavian_logo.PNG" alt="Octavian logo" width="220">
</p>

# Octavian

[Octavian](https://ghevia.github.io/octavian/) is a Python-first astrodynamics / trajectory-optimization toolkit built
on **ASSET (asset_asrl)**. 

This package includes:

- Two-impulse rendezvous with bounded free final time.
- Two-impulse rendezvous with bounded variable pre-coast and final elapsed
  mission time, including support for a zero pre-coast lower bound.
- Composable coast phases with continuous or impulsive links.
- Terminal state, terminal delta-v, path, and orbital-element constraints.
- Finite chemical-burn phases with mass depletion and three thrust-direction controls.
- J2, Sun/Moon gravity, exponential cannonball drag, and cannonball SRP in
  inertial and exact relative dynamics.
- Central-body selection, CWH relative motion, RIC transforms and plots,
  differential relative perturbations, and SPICE solar-phase geometry.
- Finite-thrust and low-thrust phases with mass depletion and spiral seeding.
- Finite-thrust directions expressed as free inertial/RIC vectors, prescribed
  inertial/RIC directions, or bounded 3-2-1 Euler kinematics.
- Optional schema-versioned JSON and YAML mission configuration.
- Lambert-Izzo seed sweeps across time of flight, longway, and multi-rev cases.
- Frame-aware Plotly trajectory and time-series diagnostics with maneuver
  markers, RIC state/range/solar phase, and inertial state/elements.
- STK ``.e``, CCSDS OEM, SPICE BSP/SPK, and CSV trajectory exports with
  explicit epoch, frame, center, and object metadata.
- A consolidated `octavian.propagate` namespace for two-body, CWH, exact RIC,
  coupled relative, relative-element, and CR3BP analysis histories.
- Dimensional and canonical Earth–Moon/general CR3BP propagation, composable
  coast phases and impulsive links, direct periodic-state constraints,
  canonical/synodic conversions, Lagrange points, Jacobi diagnostics, and
  transfer/reference-orbit plotting.

`octavian` declares ASSET (`asset_asrl`) as a runtime dependency because the
solver-backed workflows rely on it.

## Install

For daily development on **Linux x86_64** (glibc 2.31 or newer), use a
repository-local virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-user -e ".[dev]"
python -m octavian.diagnostics
```

After activation, use normal commands such as
`python examples/quick/01_two_impulse_free_time.py` and `pytest`.

For local development and ASSET-backed workflows on **Windows**, use the
recommended Conda environment. It installs ASSET and its native runtime inside
one isolated prefix:

```powershell
conda env create --file environment.yml
conda run --name octavian-dev python -m pip install --no-user -e ".[dev]"
conda run --name octavian-dev python -m octavian.diagnostics
```

See the hosted [Development Environment](https://ghevia.github.io/octavian/tutorials/development-environment/)
guide for
the platform support boundary, daily commands, dependency tiers, and the
recommended environment pattern for other projects. Windows also supports a
repository-local `pip + venv` setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --no-user -e ".[dev]"
python -m octavian.diagnostics
```

For a standard installed package with the Plotly and Matplotlib backends:

```bash
pip install "octavian[viz]"
```

For solver and astrodynamics code without visualization, `pip install octavian`
is the smaller dependency set. Add YAML support only when needed:

```bash
pip install "octavian[yaml]"
```

Before running solver-backed examples, verify ASSET imports in the active
environment:

```bash
python -c "import asset_asrl; print(asset_asrl.__file__)"
```

## Examples

```bash
python examples/quick/01_two_impulse_free_time.py
python examples/quick/02_two_impulse_precoast_impulsive_link.py
python examples/composable/earth_centered/08_chemical_burn_j2.py
python examples/composable/earth_centered/12_thrust_frames_and_attitude.py
python examples/composable/earth_centered/13_cannonball_drag_srp.py
python examples/analysis/01_propagation_namespace.py
python examples/composable/relative/25_cannonball_drag_srp.py
python examples/composable/cislunar/28_canonical_periodic_orbit.py
python examples/composable/cislunar/29_periodic_orbit_transfer.py
python examples/composable/cislunar/30_high_fidelity_recapture.py
python examples/composable/cislunar/31_jacobi_targeted_periodic_orbit.py
python examples/outputs/01_ephemeris_files.py
python examples/outputs/02_matplotlib_plots.py
python -m octavian.config examples/config/01_two_impulse_transfer.json
```

Examples read sequentially from configuration to results, with short helper
functions inlined so each step can be edited in place.

The examples also save Matplotlib PNGs beside their interactive HTML plots.
Import `save_trajectory_image` (or its relative/CR3BP counterpart) from
`octavian.viz.matplotlib` and pass the trajectory, just like the HTML helpers.
Equatorial transfers and planar Lyapunov orbits use equal-scale 2D XY views;
inclined transfers and relative safety ellipses use 3D views. For examples,
output filenames, and figure customization, see the
[static plotting guide](https://ghevia.github.io/octavian/tutorials/output-files/#matplotlib-example-gallery).
Quick examples 01–04 and composable Earth-centered examples 01, 02, 03, 05,
06, 08, and 11 overlay reference orbits propagated from their boundary states.
Use `propagate.inertial(...)` for Cartesian propagation with optional J2,
third-body gravity, drag, and SRP. Other examples overlay comparison transfers.
Use the same `reference_trajectories` list with either backend; see the [reference-trajectory guide](https://ghevia.github.io/octavian/tutorials/output-files/#reference-trajectories).
Install `octavian[viz]` first; set `MPLBACKEND=Agg` when running without a display.

JSON works without another dependency. Install `octavian[yaml]` to load YAML
mission files. Both formats construct the same Python mission objects and use
the same solvers as ordinary mission scripts.

See the hosted
[example capability index](https://ghevia.github.io/octavian/examples/) for a task-oriented map of
every executable example and mission pattern.

Run every executable example with real solvers and isolated output directories:

```bash
python -m pytest tests/test_executable_examples.py -q
```

This check is included in the normal test suite and CI.

## Documentation

The docs are built with MkDocs and publish through GitHub Pages.
Browse the public documentation at
[ghevia.github.io/octavian](https://ghevia.github.io/octavian/).

```bash
pip install -e ".[dev]"
python -m mkdocs serve
python -m mkdocs build
```

When GitHub Pages is enabled for the repository with **Source: GitHub Actions**,
the docs workflow publishes the site from pushes to `dev`.

## Development

```bash
conda run --name octavian-dev python -m pip install --no-user -e ".[dev]"
conda run --name octavian-dev pre-commit install
conda run --name octavian-dev python -m build
conda run --name octavian-dev python -m pytest
```

## Studies

Use `octavian.study.grid` to run parameter sweeps and optionally save results to
disk.

## Releasing

Release versions are published from Git tags such as `v0.1.5` by GitHub Actions.
Maintainer steps are documented in
[RELEASING.md](https://github.com/GHevia/octavian/blob/dev/RELEASING.md).
