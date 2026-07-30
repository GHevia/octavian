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

The `outputs/` examples start from a solved mission and demonstrate portable
reporting artifacts such as STK ephemerides, CCSDS OEM, SPICE BSP, and CSV.

## Suggested Learning Path

The task-oriented
[`docs/examples/index.md`](../docs/examples/index.md) maps every user-facing
capability to an executable script. Use it when you know the design task but
not the API name.

Start with the quick API when a standard transfer helper already describes the
mission:

- `quick/01_two_impulse_free_time.py` — the smallest complete Octavian mission;
- `quick/02_two_impulse_precoast_impulsive_link.py` — add a departure window;
- `quick/03_time_tradeoff.py` — compare objective choices;
- `quick/04_batch_targets.py` — use ordinary Python to sweep configurations;
- `quick/05_sun_centered_transfer.py` — change the central body and scale.
- `quick/06_relative_hop.py` — compose a relative coast-burn-coast-burn hop;
- `quick/07_relative_transfer_chain.py` — string multiple relative transfers
  together with intermediate coasts.

Move to the composable API when the mission needs explicit phases and
constraints:

- `composable/earth_centered/01_single_phase_terminal_dv_objective.py` — the quick example
  expressed as vehicle, dynamics, phase, constraints, variables, and objective;
- examples 02–07 — phase links, path constraints, and orbital targeting;
- examples 08–10 — finite burns and perturbations;
- relative examples 11–12 — CWH relative motion, RIC plotting, and operational geometry;
- relative examples 13–17 — relative representations, exact dynamics formulations,
  perturbations, SPICE solar geometry, and native D'Amico targeting;
- earth-centered example 18 — low-thrust orbit raising with a dedicated initial guess;
- relative examples 18–21 — a solved safety-ellipse transfer, finite
  burn–coast–burn chain, three-burn topology, and perturbed ROE propagation;
- earth-centered example 19 — RIC-referenced thrust and bounded kinematic
  attitude states;
- earth-centered example 20 and relative example 23 — inertial and
  differential cannonball drag/SRP;
- relative example 24 — native classical relative-element propagation and
  targeting;
- cislunar examples 22 and 24–26 — dimensional CR3BP fundamentals,
  nondimensional periodic-orbit correction, an impulsive L1-to-L2 transfer,
  and perturbed inertial recapture.

After either API path, run `outputs/01_ephemeris_files.py` to see the common
ephemeris export interface.

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
