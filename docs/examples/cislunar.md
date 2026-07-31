# Designing In The Cislunar Regime

The cislunar examples form one continuous design progression:

| Step | Design model | Purpose |
| --- | --- | --- |
| 22 | Dimensional CR3BP | Learn the system, propagation, transforms, invariant, and composable phase. |
| 24 | Canonical CR3BP at the user boundary | Correct a planar L1 Lyapunov orbit with direct front/back periodicity. |
| 25 | Dimensional CR3BP inside ASSET | Coast on L1, transfer impulsively, insert at L2, and coast on L2. |
| 26 | Earth-centered ephemeris perturbation model | Re-target the nominal CR3BP endpoints under J2, Sun/Moon gravity, and SRP. |
| 27 | Canonical CR3BP target | Select a different L1 family member by Jacobi constant instead of initial x. |

This separation is intentional. Canonical CR3BP units make orbit families
easy to compare with published initial conditions; SI units keep Octavian's
mission, spacecraft, maneuver, and ephemeris interfaces consistent.

## Units At The Solver Boundary

Let one distance unit be the primary-secondary separation and one time unit be
the inverse system mean motion. A canonical state is converted explicitly:

```python
from octavian.cislunar import (
    CR3BPSystem,
    dimensionalize_state,
    dimensionalize_time,
    nondimensionalize_state,
)

system = CR3BPSystem.earth_moon()
initial_si = dimensionalize_state(initial_canonical, system)
period_s = dimensionalize_time(period_tu, system)
recovered_canonical = nondimensionalize_state(initial_si, system)
```

`Phase`, `Mission`, and `Solution` continue to use meters, meters per second,
and seconds. Example 24 performs the conversion immediately before the phase
declaration and converts the solved rows back before reporting and plotting.

## Correct A Periodic Orbit

Example 24 starts from a published-style canonical L1 seed. Its essential
ASSET declarations are:

```python
constraints=[
    constraints.periodic_state(),
    constraints.state_component("x", x0_m, where="Front"),
    constraints.state_component("y", 0.0, where="Front"),
]
```

`periodic_state()` equates the selected Cartesian components at the front and
back of the phase in the phase's native synodic frame. It does not constrain
time, so the period remains a decision variable within `tof_bounds_s`.

An autonomous periodic orbit also has an arbitrary phase shift. Fixing the
front `x` coordinate selects a family member, and the front `y=0` condition
selects a symmetry-plane crossing. The phase still has `initial_state` and
`final_state` seeds; those initialize scaling and the collocation mesh and do
not replace the periodic equality.

Use a component subset when appropriate:

```python
constraints.periodic_state(("x", "y", "vx", "vy"))
```

That is useful for a planar formulation when out-of-plane components are
separately fixed.

## Select The Orbit By Jacobi Constant

Example 27 replaces example 24's fixed initial x coordinate with a target
invariant:

```python
constraints=[
    constraints.periodic_state(),
    constraints.state_component("y", 0.0, where="Front"),
    constraints.jacobi_constant(
        3.16,
        where="Front",
        dimensional=False,
    ),
]
```

The front/back equality closes the physical state, `y=0` supplies the phase
condition, and the Jacobi target selects the orbit-family member. Initial
position, velocity, and period remain free for ASSET to correct.

`dimensional=False` accepts the order-one canonical values commonly tabulated
with nondimensional CR3BP states. The default `dimensional=True` accepts
`m²/s²`, consistent with `octavian.cislunar.jacobi_constant`. Internally the
compiler evaluates the constraint in canonical units for conditioning. An
optional `tolerance` produces symmetric upper and lower bounds; without one,
the target is an equality.

## Transfer Between Periodic Orbits

Example 25 expresses the trajectory architecture directly as three ASSET
phases:

```text
L1 orbit coast -> impulsive departure -> CR3BP transfer
              -> impulsive insertion -> L2 orbit coast
```

The two links use `links.impulsive()`. The transfer phase exposes its front
velocity jump with `ImpulsiveDeltaV(where="Front")`; the arrival coast does
the same for L2 insertion. `tof_is_relative=True` gives each phase its own
duration bounds, and `minimize_total_delta_v()` charges both maneuvers.

The plot overlays propagated L1 and L2 reference orbits, colors each solved
phase, and marks both impulses. These reference curves help interpret the
solution but do not add hidden constraints.

The supplied transfer is a deliberately compact starting problem. For
production work, replace the family seeds, add geometric path constraints,
or introduce intermediate coast/transfer phases using the same `previous`
and `link` pattern.

## Hand Off To A Perturbed Model

Example 26 demonstrates a model transition without pretending the circular
model and ephemeris model are the same system:

1. Propagate one nominal L1 orbit in the CR3BP synodic frame.
2. Sample the BSP Moon at the mission epoch and align synodic +X with the
   Earth-to-Moon direction.
3. Convert the nominal endpoints to Earth-centered inertial states.
4. Solve a second ASSET mission with Earth J2, ephemeris Moon/Sun gravity, and
   cannonball solar radiation pressure.
5. Report the boundary correction required to recapture the nominal endpoint.

The second trajectory is not mathematically periodic in an ephemeris model.
Its correction is a useful model-mismatch diagnostic and an initial guess for
a later high-fidelity design. A true ephemeris-periodic or quasi-periodic
design needs a precisely defined epoch-to-epoch boundary map, often with
additional targeting variables; Octavian does not silently substitute that
problem.

## Current Model Boundaries

- CR3BP composable phases are ballistic and use a rotating barycentric frame.
- Impulsive links between compatible CR3BP phases are supported.
- Finite-thrust CR3BP phases are not yet compiled.
- J2, third-body gravity, drag, and SRP belong to the inertial force model,
  not to the canonical CR3BP equations.
- Synodic/inertial transforms implement circular CR3BP geometry. Example 26
  explicitly aligns that geometry to the BSP Moon at the handoff epoch.
- Two-body osculating-element constraints are not meaningful as direct
  rotating-frame CR3BP constraints; use Cartesian synodic components.

These boundaries are surfaced in the examples so a design script fails
clearly instead of appearing to provide a fidelity it does not have.

## Run The Progression

```bash
python examples/composable/cislunar/22_earth_moon_cr3bp.py
python examples/composable/cislunar/24_canonical_periodic_orbit.py
python examples/composable/cislunar/25_periodic_orbit_transfer.py
python examples/composable/cislunar/26_high_fidelity_recapture.py
python examples/composable/cislunar/27_jacobi_targeted_periodic_orbit.py
```

Each script prints solver and model diagnostics and writes an interactive
trajectory plot. Examples 24–27 also write time-history diagnostics.
