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
- Lambert-Izzo seed sweeps across time of flight, longway, and multi-rev cases.
- Plotly HTML visualization with maneuver markers.

`octavian` can be installed from PyPI, but ASSET must still be installed
separately for solver-backed workflows.

## Install

```bash
pip install octavian
```

If you plan to run optimization solvers, install `asset_asrl` separately in the
same environment. In this repo's local Windows setup, solver-backed commands are
usually run through:

```bash
conda run -n asset_env python ...
```

## Examples

```bash
python examples/quick/01_two_impulse_free_time.py
python examples/quick/02_two_impulse_precoast_impulsive_link.py
python examples/composable/08_chemical_burn_j2.py
```

See the MkDocs site for tutorial-style walkthroughs of every example.

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
pip install -e ".[dev]"
pre-commit install
python -m build
pytest
```

## Studies

Use `octavian.study.grid` to run parameter sweeps and optionally save results to
disk.

## Releasing

Release versions are published from Git tags such as `v0.1.5` by GitHub Actions.
Maintainer steps are documented in [RELEASING.md](RELEASING.md).
