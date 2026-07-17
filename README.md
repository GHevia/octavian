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
- J2 perturbation support in the composable ASSET backend.
- Central-body selection, CWH relative motion, and relative safety geometry.
- Finite-thrust and low-thrust phases with mass depletion and spiral seeding.
- Optional schema-versioned JSON and YAML mission configuration.
- Lambert-Izzo seed sweeps across time of flight, longway, and multi-rev cases.
- Plotly HTML visualization with maneuver markers.

`octavian` declares ASSET (`asset_asrl`) as a runtime dependency because the
solver-backed workflows rely on it.

## Install

For local development and ASSET-backed workflows on Windows, use the supported
project environment. It installs ASSET and its native runtime inside one
isolated conda prefix:

```powershell
conda env create --file environment.yml
conda run --name octavian-dev python -m pip install --no-user -e ".[dev]"
conda run --name octavian-dev python -m octavian.diagnostics
```

See [Development Environment](docs/tutorials/development-environment.md) for
daily commands, dependency tiers, and the recommended environment pattern for
other projects.

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
python examples/composable/08_chemical_burn_j2.py
python -m octavian.config examples/config/01_two_impulse_transfer.json
```

JSON works without another dependency. Install `octavian[yaml]` to load YAML
mission files. Both formats construct the same Python mission objects and use
the same solvers as ordinary mission scripts.

See the MkDocs site for tutorial-style walkthroughs of every example and mission
pattern.

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
