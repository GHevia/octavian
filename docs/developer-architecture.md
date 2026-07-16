# Developer Architecture

This page is for contributors who need to understand where functionality lives
and how the pieces fit together. Octavian is organized around four layers:

1. Configuration: user intent and problem setup.
2. Compilation: translation into solver-compatible ASSET objects.
3. Solving: optimization execution, retries, and backend dispatch.
4. Reporting: results, summaries, persistence, and visualization.

The public API should stay Python-first: mission scripts describe what the user
wants, while backend modules decide how that intent is compiled and solved.

## Current Functionality

Octavian currently supports:

- quick two-impulse rendezvous and transfer problems,
- optional precoast before a transfer,
- composable multi-phase missions with coast, rendezvous, transfer, and
  chemical-burn modes,
- continuous and impulsive phase links,
- boundary state and position constraints,
- minimum-radius path constraints,
- terminal orbital-element constraints for semi-major axis, eccentricity, and
  inclination,
- finite chemical burns with mass depletion and thrust-direction controls,
- Earth two-body, J2, Sun, and Moon gravity perturbations in the composable
  backend,
- ASSET-backed optimization with a targeted retry path for non-monotonic mesh
  time failures,
- `Solution` and `RendezvousResult` reporting,
- Plotly trajectory visualization.

## Object Model

Octavian uses inheritance only where the relationship is truly "is-a".
Composition is preferred for "has-a" relationships and user-facing mission
assembly.

### Is-A Relationships

- `Constraint` is the abstract base class for all constraint declarations.
- `OrbitalElementConstraint` is a `Constraint` that targets an orbital element.
- `SemiMajorAxis`, `Eccentricity`, `InclinationDeg`, `MinRadius`, `State`, and
  `Position` are specific constraint declarations.

These classes share the same conceptual contract: each object describes a
solver requirement and exposes a canonical `value`.

### Has-A Relationships

- `Mission` has phases, spacecraft, objectives, solver options, a run plan, a
  retry policy, and a solve configuration.
- `Phase` has spacecraft, dynamics, constraints, variables, events, a previous
  phase, and an optional link.
- `Dynamics` has perturbation flags and physical constants.
- `Spacecraft` has thrusters and mass properties.
- `Solution` has a backend result and attempt logs.
- `RendezvousResult` has trajectory samples, maneuver markers, objective
  metadata, and backend-specific info.

These are composition relationships. For example, a phase is not a spacecraft
and is not dynamics; it owns references to those objects because they are part
of the phase definition.

## File Map

### User-Facing Configuration

- `octavian/mission.py`: top-level `Mission` container and `Mission.solve()`.
- `octavian/phase.py`: `Phase` definitions and boundary-state helper.
- `octavian/models.py`: `Dynamics`, `Perturbations`, solve config, run plans,
  stages, and retry policy.
- `octavian/spacecraft.py`: spacecraft and thruster data models.
- `octavian/constraints.py`: constraint class hierarchy and factory helpers.
- `octavian/variables.py`: user-facing optimization variables such as
  `ImpulsiveDeltaV`.
- `octavian/events.py`: boundary events such as impulses.
- `octavian/links.py`: continuous and impulsive phase link declarations.
- `octavian/objectives.py`: objective declarations such as total delta-v and
  final time.
- `octavian/conops.py`: reusable concept-of-operations mission builders.
- `octavian/quick.py`: high-level quick-start problem builders.
- `octavian/specs.py`: lower-level problem specifications used by legacy quick
  and rendezvous flows.

### Astrodynamics And Data

- `octavian/dynamics.py`: ASSET vector-function ODEs for two-body, J2,
  third-body, mass-coast, and chemical-burn dynamics.
- `octavian/astro/kepler.py`: Kepler propagation, orbital element conversion,
  and dense initial guesses.
- `octavian/astro/lambert.py`: Lambert seed generation and selection.
- `octavian/astro/types.py`: vector normalization helpers.
- `octavian/astro/units.py`: default unit scaling for ASSET phases.
- `octavian/time.py`: time-bound normalization.
- `octavian/data/ephemeris.py`: bundled reduced Sun/Moon ephemeris access,
  epoch conversion, and SPICE sampling.
- `octavian/data/parser.py`: developer utility for producing a reduced BSP from
  a larger source ephemeris. Runtime code should not depend on it.

### Solving And Compilation

- `octavian/runner.py`: validates missions, selects the backend, manages stages,
  and records retry attempts in `Solution`.
- `octavian/_asset.py`: ASSET import boundary, small compatibility helpers, and
  the shared protected solve wrapper.
- `octavian/solvers/options.py`: solver option dataclass shared by ASSET
  backends.
- `octavian/solvers/preconfigured.py`: ASSET backend for built-in,
  preconfigured two-impulse free-time and precoast-transfer specs.
- `octavian/solvers/rendezvous.py`: compatibility shim for the old
  preconfigured backend import path.
- `octavian/solvers/composable.py`: stable composable-solver entry point and
  compilation orchestrator.
- `octavian/solvers/compiler/phase_compiler.py`: phase classification, state
  dimensions, dynamics selection, guess shaping, and ASSET phase construction.
- `octavian/coordinates/`: immutable frame, state-layout, and characteristic
  scaling declarations shared by configuration, compilation, and reporting.
- `octavian/bodies/`: immutable central-body constants and case-insensitive
  catalog lookup used by quick and composable dynamics configuration.
- `octavian/solvers/constraint_compiler.py`: composable-backend constraint
  lookup, orbital-element ASSET expressions, terminal post-burn shell handling,
  and orbital-element result reports.
- `octavian/solvers/third_bodies.py`: third-body table construction and phase
  perturbation helpers used by the composable backend.

### Reporting And Visualization

- `octavian/solution.py`: stable solution wrapper, attempt log, and visualization
  namespace.
- `octavian/types.py`: shared small result types such as `Maneuver`.
- `octavian/study.py`: study-level utilities.
- `octavian/viz/plotly.py`: Plotly trajectory visualization.
- `octavian/viz/data/`: bundled visualization assets.

## Solve Flow

The normal mission flow is:

1. User builds a `Mission` with one or more `Phase` objects.
2. `Mission.solve()` creates a `MissionRunner`.
3. `MissionRunner.solve()` validates the mission and chooses a backend.
4. Quick rendezvous-like missions compile to `TwoImpulseFreeTimeSpec` or
   `TwoImpulsePreCoastSpec` and run through `octavian.solvers.preconfigured`.
5. Missions with explicit composable features, perturbations, variables, or
   chemical burns run through `octavian.solvers.composable`.
6. Backends construct ASSET OCPs and call
   `octavian._asset.solve_with_standard_sequence()`.
7. The backend extracts a `RendezvousResult`.
8. The runner wraps the result in `Solution` with attempt metadata.

## ASSET Failure Handling

`octavian._asset.solve_with_standard_sequence()` is the shared solve boundary
for ASSET-backed solvers. It calls `solve_optimize_solve()` once using the
configured mesh settings. ASSET raises the known
`Non monotonic time coordinates in LGLInterpTable.` exception from
`LGLInterpTable::checkInput()` when adjacent table times reverse relative to the
overall table direction. Octavian catches that diagnostic and retries the same
OCP with adaptive mesh disabled on the OCP and all compiled phases passed by the
solver.

If the retry fails with the same diagnostic, Octavian raises
`AssetNonMonotonicTimeError` with a clearer message. The message recommends
coarser mesh settings, wider time bounds, or a simpler initial guess. Other
ASSET errors are not caught or reclassified.

This wrapper is intentionally narrow. It protects the most common hard mesh
failure without hiding unrelated optimizer failures.

## Current Design Seams

The composable compiler is being split by compilation responsibility. Phase
classification, phase dimensions, dynamics selection, mass/burn guess shaping,
and ASSET phase construction now live in
`octavian/solvers/compiler/phase_compiler.py`. Private aliases remain in
`octavian.solvers.composable` during the transition so existing internal tools
and focused tests do not break.

The orchestrator still owns several responsibilities:

- guess construction,
- objective compilation,
- phase linking,
- result extraction.

The first split is already in place: constraint lookup, orbital-element
constraint compilation, terminal post-burn shell handling, and constraint
reporting live in `octavian/solvers/constraint_compiler.py`.

Future restructuring should continue splitting by compiler responsibility
rather than by arbitrary helper buckets. The next candidate modules are:

- `solvers/compiler/guessing.py` for Kepler, Lambert, and powered-arc seeds,
- `solvers/compiler/objective_compiler.py` for objective normalization and ASSET costs,
- `solvers/compiler/result_extraction.py` for trajectories, maneuvers, chemical-burn
  summaries, and constraint reports.

Avoid creating a broad `utils.py` for unrelated helpers. Shared code should move
to a module named after the concept it owns, such as `_asset.py` for ASSET
compatibility or `third_bodies.py` for third-body table construction.

## Contributor Rules Of Thumb

- Use inheritance for shared contracts and true subtype relationships.
- Use composition for mission assembly and solver configuration.
- Keep user intent objects free of ASSET objects.
- Keep ASSET-specific code behind `_asset.py` or solver modules.
- Prefer small dataclasses with explicit names over dictionaries when the
  structure is part of the API.
- Do not make a helper public until more than one module needs the same concept.
- Update examples and docs in the same PR as user-facing behavior changes.

## Robustness Campaigns

Solver robustness tests live under `tests/robustness`. The orbit-transfer
campaign generates valid elliptic endpoint orbits from a fixed master seed and
stores a separate seed for every case. This makes any optimizer failure
reproducible without relying on global NumPy random state.

The regular solver suite runs a small representative sample. Every fifth
scenario transfers between low orbit and an endpoint at least 500 km above
GEO. Raising and lowering cases exercise both quick and composable backends
across direct, continuous-link, and impulsive-link layouts. Run the larger
campaign sequentially in the ASSET conda environment with:

```powershell
conda run -n asset_env python tests/robustness/run_orbit_transfer_campaign.py --cases 100
```

The runner writes a JSON report containing every scenario, selected API knobs,
runtime, result metrics, and exception details. To expand the pytest sample
without using the report runner, set `OCTAVIAN_ROBUSTNESS_CASES` before invoking
pytest. Keep the master seed fixed for release qualification; use a different
seed for exploratory campaigns and preserve any failing case seed as a
regression test.
