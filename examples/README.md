# Octavian Examples

Every example is a flat, executable mission script. Treat one as a starting
configuration: copy it, change the states or mission declarations, and run the
file directly. The examples intentionally avoid application-style entry-point
guards so the mission reads from top to bottom in the same order Octavian uses
it:

1. define the physical constants and boundary states;
2. configure the vehicle, dynamics, phases, and objectives;
3. solve the mission;
4. inspect and plot the result.

## Suggested Learning Path

Start with the quick API when a standard transfer helper already describes the
mission:

- `quick/01_two_impulse_free_time.py` — the smallest complete Octavian mission;
- `quick/02_two_impulse_precoast_impulsive_link.py` — add a departure window;
- `quick/03_time_tradeoff.py` — compare objective choices;
- `quick/04_batch_targets.py` — use ordinary Python to sweep configurations;
- `quick/05_sun_centered_transfer.py` — change the central body and scale.

Move to the composable API when the mission needs explicit phases and
constraints:

- `composable/01_single_phase_terminal_dv_objective.py` — the quick example
  expressed as vehicle, dynamics, phase, constraints, variables, and objective;
- examples 02–07 — phase links, path constraints, and orbital targeting;
- examples 08–10 — finite burns and perturbations;
- examples 11–12 — relative motion and operational geometry;
- example 13 — low-thrust orbit raising with a dedicated initial guess.

All mission inputs use SI units unless a parameter name explicitly says
otherwise. Generated Plotly HTML files are written to the current directory.

## Literal Config Files

`config/01_two_impulse_transfer.json` expresses the same first composable
mission without Python syntax. Run it with:

```bash
python -m octavian.config examples/config/01_two_impulse_transfer.json
```

JSON and YAML use the same versioned schema and construct the same public
`Mission`, `Phase`, and model objects as Python scripts. Python remains the
most expressive interface; config files are useful when a plain declarative
artifact is easier to generate, review, or exchange.
