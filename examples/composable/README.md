# Composable examples

The composable examples are grouped by the state and frame used by the
mission:

- `earth_centered/` contains ordinary Earth-centered inertial transfers,
  perturbation demonstrations, and powered-flight examples.
- `relative/` contains chief-centered RIC missions, relative representations,
  relative-element propagation, and relative finite-burn examples.

The numeric prefixes preserve the tutorial build-up order documented in
`docs/examples/composable.md`.

Earth-centered example 19 introduces frame-aware finite-thrust controls and
bounded Euler-angle kinematics.

Relative example 23 introduces spacecraft-specific cannonball properties,
exponential drag, and BSP-driven solar radiation pressure.
