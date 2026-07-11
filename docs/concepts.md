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
rendezvous, and chemical-burn phases.

Common phase inputs:

- `mode`: the phase type, such as `coast` or `chemical_burn`.
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

Objectives define what the optimizer minimizes. The current scripts primarily
use total delta-v and, in the quick API, an optional final-time weight.

## Dynamics And Perturbations

`Dynamics` configures the gravitational parameter, central-body radius, J2
coefficient, and perturbation flags. J2, Moon, and Sun perturbations are
implemented in the composable ASSET backend for coast and chemical-burn phases.
Moon and Sun use the bundled reduced DE440 ephemeris in the `ECI_TOD` frame and
require a mission initial epoch so Octavian can build ASSET interpolation
tables over the mission time bounds.

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

Plotly helpers turn the trajectory and maneuvers into inspectable HTML files.
