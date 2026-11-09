# Octavian Examples

Every example is an executable mission or analysis script. Treat one as a
starting configuration: copy it, change the states or declarations, and run
the file directly. Lettered companions (such as earth-centered 11b) stay
beside their numbered topic in the same folder. The examples intentionally avoid application-style
entry-point guards so the workflow reads from top to bottom.

1. define the physical constants and boundary states;
2. configure the vehicle, dynamics, phases, and objectives;
3. solve the mission;
4. inspect and plot the result.

The `outputs/` examples start from a solved mission and demonstrate portable
reporting artifacts such as STK ephemerides, CCSDS OEM, SPICE BSP, and CSV.
Quick example 08 demonstrates propagators and conversions without a solve.
Earth-centered examples 14–15 demonstrate perturbed propagation and direct ASSET ODEs.

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
- `composable/earth_centered/11b_low_thrust_plane_change.py` — Q-law seeding
  for a 100 mN small-satellite orbit raise with inclination change, throttle
  coloring, and firing intervals;
- `composable/earth_centered/11c_low_thrust_perturbation_continuation.py` — solve
  a 100 mN transfer without perturbations, then reuse its full solved history
  through J2, 8×8 Earth gravity, and Sun/Moon continuation stages;
- earth-centered examples 11–13 — low-thrust orbit raising, thrust-attitude
  representations, and cannonball drag/SRP;
- earth-centered examples 14–15 — 8×8 EGM2008 with Sun/Moon gravity and model comparison;
- relative examples 16–17 — CWH relative motion, RIC plotting, and operational geometry;
- relative examples 18–22 — relative representations, exact dynamics formulations,
  perturbations, SPICE solar geometry, and native D'Amico targeting;
- relative examples 23–26 — a solved safety-ellipse transfer, finite
  burn–coast–burn chain, three-burn topology, and perturbed ROE propagation;
- relative example 27 — differential cannonball drag/SRP;
- relative example 28 — native classical relative-element propagation and
  targeting;
- cislunar examples 29–34 — dimensional CR3BP fundamentals,
  nondimensional periodic-orbit correction, an impulsive L1-to-L2 transfer,
  perturbed inertial recapture, and Jacobi-targeted family selection.

For analysis without a solve, run
`quick/08_propagation_models.py` to compare the two-body, CWH, nonlinear
RIC, coupled relative, relative-element, and CR3BP entry points.

Mission inputs use SI units unless an API explicitly selects another
convention, such as `Dynamics.cr3bp(dimensional=False)`. Generated Plotly HTML
and Matplotlib PNG files are written to the current directory. Install
`octavian[viz]` for both backends. Use `MPLBACKEND=Agg` for headless image export.

The [Matplotlib example gallery](../docs/tutorials/output-files.md#matplotlib-example-gallery)
lists static outputs: equatorial transfers and planar Lyapunov orbits in 2D,
and inclined transfers and relative safety ellipses in 3D. The cislunar PNGs
in examples 31 and 33 zoom to the orbits so their geometry stays readable;
the companion HTML files retain the Earth–Moon overview.

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

## Validate Every Example

Run the executable examples with real solvers and headless plots:

```bash
python -m pytest tests/test_executable_examples.py -q
```

The test discovers every Python example and JSON mission, runs each in a
separate process and temporary output directory, and reports failures with
captured logs. It is also part of the normal test suite used by CI. Install
`octavian[dev]` in the supported development environment first.
