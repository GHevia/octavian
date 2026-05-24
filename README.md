# Octavian

Octavian is a Python-first astrodynamics / trajectory-optimization toolkit built on **ASSET (asset_asrl)**.

This MVP includes:
- Two-impulse rendezvous with bounded free final time (single coast phase)
- Two-impulse rendezvous with bounded **variable pre-coast** (two phases + link Δv objective)
- Finite chemical-burn phases with mass depletion and three thrust-direction controls
- J2 perturbation support in the composable ASSET backend
- Lambert-izzo seed sweeps across TOF / longway / multi-rev (ASSET's `Astro.lambert_izzo`)
- Fast two-body initial guesses via ASSET Kepler propagation (`Astro.propagate_cartesian`)
- Plotly HTML visualization with maneuver markers

`octavian` can be installed from PyPI, but ASSET must still be installed separately for solver-backed workflows.

## Install

```bash
pip install octavian
```

If you plan to run the optimization solvers, install `asset_asrl` separately in the same environment.

## Examples

```bash
python examples/quick/01_two_impulse_free_time.py
python examples/quick/02_two_impulse_precoast_impulsive_link.py
python examples/composable/10_chemical_burn_j2.py
```


## Development

```bash
pip install -e ".[dev]"
pre-commit install
python -m build
pytest
```

## Studies

Use `octavian.study.grid` to run parameter sweeps and optionally save results to disk.

## Releasing

Release versions are published from Git tags such as `v0.1.3` by GitHub Actions. Maintainer steps are documented in [RELEASING.md](RELEASING.md).
