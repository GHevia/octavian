# Composable examples

The composable examples are grouped by the state and frame used by the
mission:

- `earth_centered/` contains ordinary Earth-centered inertial transfers,
  perturbation demonstrations, and powered-flight examples.
- `relative/` contains chief-centered RIC missions, relative representations,
  relative-element propagation, and relative finite-burn examples.
- `cislunar/` contains rotating-frame CR3BP propagation and optimization
  examples.

The numeric prefixes preserve the tutorial build-up order documented in
`docs/examples/composable.md`.

For a task-oriented map of every script, use `docs/examples/index.md`.

## Current Progressions

| Regime | Progression |
| --- | --- |
| Earth-centered | Transfer basics (01–07), finite burns and perturbations (08–10), low thrust and attitude (18–19), cannonball drag/SRP (20). |
| Relative | CWH and geometry (11–12), representations and exact dynamics (13–16), ROE targeting and multiphase missions (17–21), differential forces and classical ROEs (23–24). |
| Cislunar | Dimensional CR3BP fundamentals (22), canonical periodic-orbit correction (24), L1-to-L2 transfer (25), perturbed inertial recapture (26). |

The cislunar sequence is explained in detail in
`docs/examples/cislunar.md`, including the canonical/SI boundary and current
model limitations.
