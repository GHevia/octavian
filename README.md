# Octavian

Octavian is a Python-first astrodynamics / trajectory-optimization toolkit built on **ASSET (asset_asrl)**.

This MVP includes:
- Two-impulse rendezvous with bounded free final time (single coast phase)
- Two-impulse rendezvous with bounded **variable pre-coast** (two phases + link Δv objective)
- Lambert-izzo seed sweeps across TOF / longway / multi-rev (ASSET's `Astro.lambert_izzo`)
- Fast two-body initial guesses via ASSET Kepler propagation (`Astro.propagate_cartesian`)
- Plotly HTML visualization with maneuver markers

ASSET must be installed separately.

## Install

```bash
pip install -e .
pip install "octavian[viz]"
```

## Examples

```bash
python examples/run_two_impulse_free_time.py
python examples/run_two_impulse_precoast.py
```
