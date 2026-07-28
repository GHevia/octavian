# Concepts

Octavian keeps the user-facing mission script close to the engineering intent.
The key objects map to different parts of the optimization workflow.

## Mission

A `Mission` is the top-level problem definition. It owns phases, objectives,
solver options, and seed-search settings. Calling `mission.solve()` validates
the mission, selects the backend, builds solver-compatible structures, and
returns a `Solution`.

## Phase

A `Phase` describes one segment of flight. Current examples use coast,
rendezvous, relative-coast, and finite-thrust phases.

Common phase inputs:

- `mode`: the phase type, such as `coast`, `finite_thrust`, or the compatible
  chemical-specific spelling `chemical_burn`.
- `spacecraft`: mass and thruster configuration.
- `dynamics`: gravity and perturbation configuration.
- `tof_bounds_s`: allowed time-of-flight bounds.
- `constraints`: fixed states, path constraints, or terminal orbital elements.
- `variables`: decision variables such as impulsive delta-v.
- `previous` and `link`: phase ordering and boundary semantics.

## Constraints

Constraints encode requirements the solver should satisfy. The current examples
show:

- Full Cartesian boundary states.
- Minimum-radius path constraints.
- Terminal semi-major axis.
- Terminal eccentricity.
- Terminal inclination.
- Offset spherical keep-out zones.
- One-sided approach cones.
- Fixed-direction lighting-angle bounds.

## Variables

Variables expose degrees of freedom. The most common user-facing variable is
`ImpulsiveDeltaV`, which lets the compiler relax velocity at a phase boundary
and charge the velocity jump in the objective.

## Links

Links define what happens at a phase boundary.

`links.continuous()` keeps position, velocity, and time continuous.

`links.impulsive()` keeps position and time continuous while allowing velocity
to jump. That jump becomes an impulsive maneuver when an appropriate
`ImpulsiveDeltaV` variable is declared.

## Objectives

Objectives define what the optimizer minimizes. Use total delta-v for declared
impulsive maneuvers, propellant for finite-thrust phases, and total time for
time-of-flight trade studies. Propellant is evaluated once at the final
powered mass state, so a burn-coast-burn chain is not double-counted.

## Dynamics And Perturbations

`Dynamics` configures the gravitational parameter, central-body radius, J2
coefficient, reference frame, characteristic scaling, and perturbation flags.
J2, Moon, and Sun perturbations are
implemented in the composable ASSET backend for coast and finite-thrust phases.
Moon and Sun use the bundled reduced DE440 ephemeris in the `ECI_TOD` frame and
require a mission initial epoch so Octavian can build ASSET interpolation
tables over the mission time bounds.

Built-in `EARTH`, `MOON`, and `SUN` definitions keep gravity, radius, J2, and
frame origin consistent. Use `Dynamics.for_body(SUN)` for composable missions
or pass `central_body=SUN` to the quick transfer builder. Raw gravitational
parameters remain supported for custom bodies and backward compatibility.

Sun-centered support currently means idealized heliocentric two-body dynamics.
It does not yet generate planetary ephemeris states or model sphere-of-influence
departure and arrival transitions.

### Powered Phases

`mode="finite_thrust"` selects the propulsion-neutral powered equations of
motion: Cartesian position and velocity, spacecraft mass, and a three-component
vector throttle. Its norm is bounded by one, and mass flow follows the selected
thruster's thrust and specific impulse. `mode="chemical_burn"` remains a fully
supported compatibility spelling and is reported as a chemical burn.

Powered phases can be used alone or in arbitrary powered/coast sequences. Coast
phases between the first and last powered phases automatically carry the mass
state so continuous links conserve spacecraft mass. A chain uses one spacecraft
configuration, but each powered phase may select a named thruster with
`phase.info["thruster"]`.

Use `objectives.minimize_propellant()` to maximize the final mass of the chain.
The generic `powered_phases` result table reports mass use for every powered
mode; the older `chemical_burns` key remains available for chemical phases.

`mode="low_thrust"` uses the same physical ODE and identifies the phase for
low-thrust-specific seed construction and reporting. The built-in
`guesses.low_thrust_spiral(...)` seed integrates constant-throttle tangential
steering with mass depletion, then releases every control sample and phase time
to the optimizer. Its `auto` direction raises or lowers based on the final
Cartesian seed anchor's radius.

The spiral seed is intended for near-circular, approximately coplanar orbit
raising or lowering. It is not an analytical solution and does not constrain
the optimized steering law. Use terminal orbital-element constraints so final
orbital phase remains free. Highly eccentric transfers, large plane changes,
and interplanetary low-thrust arcs will need additional seed families.

### Relative Motion

Relative states use the chief's RIC/RTN/LVLH axes: R is radial, I is in-track,
and C is cross-track. Two dynamics levels are deliberately separate:

- `Dynamics.cwh(...)` is the unforced, linear Clohessy-Wiltshire model for a
  circular chief and small separation. It is useful for quick studies and fast
  initial guesses. Adding perturbations to it is rejected.
- `Dynamics.relative(chief_initial_state_eci=...)` is the full nonlinear
  model. The optimizer propagates independent chief and deputy absolute
  Cartesian states under central gravity and any enabled J2, Moon, or Sun
  perturbations. Constraints, maneuvers, results, and plots are converted to
  instantaneous RIC at the compiler boundary. The propagated absolute
  histories remain available as `solution.chief_trajectory_eci` and
  `solution.deputy_trajectory_eci`.

`octavian.relative` includes explicit single-state and vectorized history
transforms:

- `absolute_to_relative_state(...)` and `relative_to_absolute_state(...)`;
- `absolute_to_relative_history(...)` and `relative_to_absolute_history(...)`;
- `ric_basis(...)` and the compatibility name `lvlh_basis(...)`.
- `absolute_to_relative_orbital_elements(...)` and
  `relative_orbital_elements_to_absolute_state(...)` for six
  quasi-nonsingular elements `[δa, δλ, δex, δey, δix, δiy]`.

The transforms include the RIC angular-rate term in velocity. When chief
acceleration is supplied, they also include orbit-plane rotation from
cross-track acceleration. Rotating an ECI velocity difference by the position
direction-cosine matrix alone is not a valid relative velocity.

For analysis that should propagate the chief rather than prescribe it,
`propagate_relative_numerical(...)` advances absolute chief and deputy states
together with central gravity, J2, lunar gravity, and solar gravity. Its result
contains both absolute histories and the converted RIC history. It uses a
fixed-step fourth-order Runge-Kutta integrator, so `max_step_s` is an explicit
accuracy/cost choice rather than a hidden tolerance.

The relative compiler currently supports one optimized relative phase. Inertial
orbital-element constraints, finite thrust, and inertial/relative phase links
remain rejected until an explicit acceleration or frame-link model is
configured.

Relative geometry constraints operate on Cartesian position in the phase
frame. `keep_out_sphere` accepts an arbitrary center, `approach_cone` defines a
forward axis and half-angle, and `lighting_angle` bounds the angle to a fixed
direction. `solar_phase_angle` instead samples the Sun from the bundled SPICE
BSP at `Mission.initial_epoch`, subtracts the chief position, and rotates the
line into RIC throughout the arc. It requires `chief_initial_state_eci` so the
rotation is defined. Neither lighting constraint is an eclipse or power model.

## Frames, Layouts, And Scaling

Every `Dynamics` declaration carries a `CoordinateFrame`; existing missions
default to Earth-centered inertial coordinates. Solver results preserve that
metadata through `solution.frame`, so trajectory consumers do not have to infer
an origin or orientation from array shape.

The compiler uses named `StateLayout` objects for position, velocity, mass, and
controls. Public trajectories remain `[R, V, t]`, while powered phases can carry
mass and thrust controls without spreading raw column numbers through compiler
passes.

Octavian still accepts and returns SI values. `SolverScaling` only declares the
characteristic units used to condition the optimization problem. Automatic
endpoint-derived scaling remains the default, and the selected units are
available through `solution.scaling`.

## Spacecraft And Thrusters

`Spacecraft` and `Thruster` hold mass and propulsion configuration. Impulsive
examples only need a lightweight thruster placeholder. Chemical-burn examples
use thrust, specific impulse, and propellant mass to propagate mass depletion.

## Solution

`Solution` wraps the backend result and attempt logs. A converged result can
report:

- final time,
- total delta-v,
- maneuver list,
- trajectory samples,
- phase segments,
- chemical burn summaries,
- constraint reports.
- reference-frame and solver-scaling metadata.
- dynamics-model metadata for relative trajectories.

Plotly helpers turn the trajectory and maneuvers into inspectable HTML files.
`save_relative_trajectory_html(...)` labels R, I, and C explicitly, places the
chief at the origin, and can draw a chief or keep-out radius.
`solution.viz().save_html(...)` selects that view automatically when the result
frame is relative. `solution.viz().save_diagnostics_html(...)` writes shared
time-axis state and geometry panels. Relative diagnostics include RIC state,
range, speed, and solar phase angle when ephemeris geometry is present;
inertial diagnostics include Cartesian state, radius, speed, and osculating
elements.

## Documentation Contract

Octavian treats docs and examples as part of the product surface. If a new
capability changes how users configure, solve, or inspect missions, the same PR
should update the relevant tutorial, example guide, and API docstrings. A
feature that exists only in code is not complete enough for users to rely on.
