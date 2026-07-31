<p align="center">
  <img src="assets/octavian_logo.PNG" alt="Octavian logo" width="220">
</p>

# Octavian

Octavian is a Python-first astrodynamics / trajectory-optimization toolkit built
on **ASSET (asset_asrl)**.

This MVP includes:

- Two-impulse rendezvous with bounded free final time.
- Two-impulse rendezvous with bounded variable pre-coast.
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

See [Development Environment](docs/tutorials/development-environment.md) for
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

For a standard installed package with the Plotly examples:

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
python examples/composable/earth_centered/19_thrust_frames_and_attitude.py
python examples/composable/earth_centered/20_cannonball_drag_srp.py
python examples/analysis/01_propagation_namespace.py
python examples/composable/relative/23_cannonball_drag_srp.py
python examples/composable/cislunar/24_canonical_periodic_orbit.py
python examples/composable/cislunar/25_periodic_orbit_transfer.py
python examples/composable/cislunar/26_high_fidelity_recapture.py
python examples/composable/cislunar/27_jacobi_targeted_periodic_orbit.py
python examples/outputs/01_ephemeris_files.py
python -m octavian.config examples/config/01_two_impulse_transfer.json
```

JSON works without another dependency. Install `octavian[yaml]` to load YAML
mission files. Both formats construct the same Python mission objects and use
the same solvers as ordinary mission scripts.

See the MkDocs
[example capability index](docs/examples/index.md) for a task-oriented map of
every executable example and mission pattern.

## Documentation

The docs are built with MkDocs and publish through GitHub Pages.

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
Maintainer steps are documented in [RELEASING.md](RELEASING.md).
