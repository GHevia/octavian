# Composable examples

The composable examples are grouped by the state and frame used by the
mission:

- `earth_centered/` contains ordinary Earth-centered inertial transfers,
  perturbation demonstrations, and powered-flight examples.
- `relative/` contains chief-centered RIC missions, relative representations,
  relative-element propagation, and relative finite-burn examples.
- `cislunar/` contains rotating-frame CR3BP propagation and optimization
  examples.

The numeric prefixes are globally unique and contiguous within each regime,
matching the grouped tutorial order in `docs/examples/composable.md`.

For a task-oriented map of every script, use `docs/examples/index.md`.

## Current Progressions

| Regime | Progression |
| --- | --- |
| Earth-centered | Transfer basics (01–07), finite burns and perturbations (08–10), low thrust, attitude, and cannonball forces (11–13). |
| Relative | CWH and geometry (14–15), representations and exact dynamics (16–19), ROE targeting and multiphase missions (20–24), differential forces and classical ROEs (25–26). |
| Cislunar | Dimensional CR3BP fundamentals (27), canonical periodic-orbit correction (28), L1-to-L2 transfer (29), perturbed inertial recapture (30), Jacobi-targeted family selection (31). |

The cislunar sequence is explained in detail in
`docs/examples/cislunar.md`, including the canonical/SI boundary and current
model limitations.
