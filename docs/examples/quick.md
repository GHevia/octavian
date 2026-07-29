# Quick API Examples

The quick examples use high-level builders for common inertial and relative
impulsive workflows. They are the best starting point when you want a compact
mission script and do not need to manage every phase yourself.

The files are deliberately flat scripts rather than importable application
modules. Read them from constants to states to mission to solution, then copy
one and edit it as a Python configuration file.

## Shared Quick API Terms

| Setting | What it controls |
| --- | --- |
| `initial_state`, `target_state` | Initial and target Cartesian boundary states created with `state(...)`. |
| `mu_m3ps2` | Raw central-body gravitational parameter for backward-compatible or custom-body use. |
| `central_body` | A consistent body preset such as `EARTH`, `MOON`, or `SUN`; body constants override `mu_m3ps2`. |
| `tf_bounds_s` | Allowed final transfer time. This is absolute mission time in the quick API. |
| `nsegs` | Mesh segment count for the optimized transfer. |
| `lambert_grid_size` | Number of time-of-flight samples used while finding Lambert initial guesses. |
| `nrevs_to_try` | Lambert revolution counts to consider. Use `(0,)` for the simplest transfer family. |
| `solver_options` | ASSET solver controls such as `print_level`; pass `SolverOptions(asset_threads=(1, 1))` for deterministic single-threaded ASSET solves in regression tests. |
| `name` | Label used in solution summaries and plot titles. |

Relative quick builders use RIC boundary states. They accept a chief ECI state,
one or more relative targets, exact or CWH dynamics, optional perturbations,
and duration bounds for every coast and transfer.

## 01: Hohmann Transfer Between Circular Orbits

Path: `examples/quick/01_two_impulse_free_time.py`

Run:

```bash
python examples/quick/01_two_impulse_free_time.py
```

Capability showcased:

- A minimal high-level transfer solve.
- Two impulsive maneuvers.
- Free final time within bounds.
- A circular-orbit target with an analytical Hohmann reference used in tests.

Important choices:

| Code | Purpose |
| --- | --- |
| `R_INITIAL_M = 7_000e3` | Sets the departure circular orbit radius. |
| `R_FINAL_M = 12_000e3` | Sets the arrival circular orbit radius. |
| `target_state` at `[-R_FINAL_M, 0, 0]` | Places the target opposite the start so the transfer resembles a Hohmann half-ellipse. |
| `tf_bounds_s=(3_000.0, 7_000.0)` | Brackets the expected Hohmann transfer time without forcing it exactly. |
| `nsegs=60` | Gives a moderate mesh for a smooth coast arc. |
| `lambert_grid_size=60` | Searches enough Lambert seeds to robustly initialize the optimizer. |
| `nrevs_to_try=(0,)` | Keeps the example focused on the direct zero-revolution solution. |

Expected output:

- Printed solution summary.
- `traj_quick_hohmann_transfer.html`.

Screenshot placeholder: `docs/assets/screenshots/quick-01-hohmann-transfer.png`.

## 02: Precoast Plus Circular-Orbit Transfer

Path: `examples/quick/02_two_impulse_precoast_impulsive_link.py`

Run:

```bash
python examples/quick/02_two_impulse_precoast_impulsive_link.py
```

Capability showcased:

- A bounded loiter before departure.
- A two-phase quick mission.
- Impulsive link behavior without writing the composable phase objects manually.

Important choices:

| Code | Purpose |
| --- | --- |
| `precoast=True` | Builds a precoast phase before the transfer phase. |
| `t1_bounds_s=(1.0, 1_000.0)` | Bounds the precoast end time. The lower bound avoids a degenerate zero-duration phase. |
| `tf_bounds_s=(1_200.0, 12_000.0)` | Keeps final time after the precoast while allowing the optimizer room to choose transfer duration. |
| `precoast_grid_size=12` | Samples twelve candidate departure times before optimization. |
| `lambert_grid_size=50` | Searches transfer seeds for each sampled precoast candidate. |
| `nrevs_to_try=(0,)` | Keeps the example on the direct transfer family. |

Expected output:

- Printed solution summary.
- `traj_quick_precoast_circular_transfer.html`.

Screenshot placeholder: `docs/assets/screenshots/quick-02-precoast-circular-transfer.png`.

## 03: Delta-v Versus Time Trade

Path: `examples/quick/03_time_tradeoff.py`

Run:

```bash
python examples/quick/03_time_tradeoff.py
```

Capability showcased:

- Objective trade studies.
- Same geometry solved twice.
- Delta-v-only behavior compared with a time-weighted solve.

Important choices:

| Code | Purpose |
| --- | --- |
| `missions = [("dv_only", 0.0), ("dv_plus_time", 2.0)]` | Runs the same target with two objective weights. |
| `w_time=0.0` | Minimizes total delta-v only. |
| `w_time=2.0` | Adds a final-time penalty to bias the optimizer toward shorter transfers. |
| `tf_bounds_s=(600.0, 20_000.0)` | Gives both objective settings a broad time window. |
| output tag in filename | Keeps the two result plots separate for comparison. |

Expected output:

- Two printed solution summaries.
- `traj_quick_time_tradeoff_dv_only.html`.
- `traj_quick_time_tradeoff_dv_plus_time.html`.

Screenshot placeholders:

- `docs/assets/screenshots/quick-03-dv-only.png`.
- `docs/assets/screenshots/quick-03-dv-plus-time.png`.

## 04: Batch Target-Radius Sweep

Path: `examples/quick/04_batch_targets.py`

Run:

```bash
python examples/quick/04_batch_targets.py
```

Capability showcased:

- Parameter sweeps in plain Python.
- Repeated mission construction.
- Selecting the best converged solution by total delta-v.

Important choices:

| Code | Purpose |
| --- | --- |
| `target_radii_m = np.linspace(8_000e3, 14_000e3, 7)` | Defines seven circular target radii to test. |
| loop over `target_radii_m` | Builds and solves one mission per target. |
| `precoast=True` | Allows each target case to choose departure timing. |
| `nrevs_to_try=(0,)` | Keeps the sweep fast and comparable. |
| `best = min(converged, key=lambda x: x[0])` | Selects the converged case with lowest total delta-v. |

Expected output:

- Per-case printed summary lines.
- Best-case summary.
- `traj_quick_batch_best.html`.

Screenshot placeholder: `docs/assets/screenshots/quick-04-batch-best.png`.

## 05: Idealized Sun-Centered Transfer

Path: `examples/quick/05_sun_centered_transfer.py`

Run:

```bash
python examples/quick/05_sun_centered_transfer.py
```

Capability showcased:

- Selecting the built-in `SUN` central body.
- An idealized circular Earth-orbit to circular Mars-orbit Hohmann transfer.
- Heliocentric frame and characteristic scaling metadata in the solution.

The endpoints are synthetic circular heliocentric states. This example does not
yet use planetary ephemerides or patch planet-centered departure and arrival
arcs.

## 06: Relative Hop

Path: `examples/quick/06_relative_hop.py`

Run:

```bash
python examples/quick/06_relative_hop.py
```

Capability showcased:

- A coast before the relative transfer.
- Impulsive departure and arrival burns.
- Exact coupled chief/deputy propagation in the quick API.
- Differential J2 enabled with `Perturbations(j2=True)`.
- RIC trajectory and time-history diagnostic plots.

`relative_hop(...)` returns an ordinary `Mission`. You can inspect or edit its
two phases before solving when a quick design needs one additional constraint
or variable.

Expected outputs:

- `traj_quick_relative_hop.html`.
- `diagnostics_quick_relative_hop.html`.

## 07: Chained Relative Transfers

Path: `examples/quick/07_relative_transfer_chain.py`

Run:

```bash
python examples/quick/07_relative_transfer_chain.py
```

Capability showcased:

- Two relative transfers in one optimization problem.
- An exact post-arrival RIC state at the first target.
- A bounded natural coast before departure for the next target.
- Four impulses: depart, arrive, depart, arrive.
- Per-transfer and per-coast duration bounds.

Add more targets to the `target_states_ric` list to extend the chain. A single
time-bound pair is repeated, while a list of pairs configures each segment
individually.

Expected outputs:

- `traj_quick_relative_transfer_chain.html`.
- `diagnostics_quick_relative_transfer_chain.html`.

## When to Leave the Quick API

Use the composable API when the mission needs explicit phase boundaries,
continuous links, finite burns, path constraints, native relative-element
targeting, an arbitrary burn topology, or custom per-phase behavior. The
relative quick builders already support J2, Sun, and Moon perturbations through
exact coupled propagation.
