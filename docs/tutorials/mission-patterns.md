# Mission Patterns

This tutorial shows how to use the current Octavian options as building blocks.
Start with the quick API when the mission is a standard two-impulse transfer.
Move to the composable API when you need explicit phases, links, custom
constraints, finite burns, or perturbations.

## Choose the API Layer

Use `two_burn_rendezvous` when your mission is:

- one transfer from `x0` to `xf`,
- optionally preceded by a bounded precoast,
- solved with impulsive departure and arrival maneuvers,
- seeded by Lambert guesses.

Use `Mission` and `Phase` directly when you need:

- more than one explicitly named phase,
- a continuous or impulsive link between phases,
- path constraints such as minimum radius,
- terminal orbital-element constraints,
- finite chemical-burn phases,
- J2 perturbations,
- custom objectives or per-phase mesh settings.

## Pattern 1: Standard Two-Impulse Transfer

```python
mission = two_burn_rendezvous(
    x0,
    xf,
    mu_m3ps2=MU,
    tf_bounds_s=(3_000.0, 7_000.0),
    nsegs=60,
    lambert_grid_size=60,
    nrevs_to_try=(0,),
)
```

Use this when final time is unknown but bounded. Increase
`lambert_grid_size` if the transfer has a narrow feasible window. Add
additional revolution counts with `nrevs_to_try=(0, 1)` when multi-rev
solutions are physically plausible and useful.

## Pattern 2: Add a Precoast

```python
mission = two_burn_rendezvous(
    x0,
    xf,
    precoast=True,
    t1_bounds_s=(1.0, 1_000.0),
    tf_bounds_s=(1_200.0, 12_000.0),
    precoast_grid_size=12,
    lambert_grid_size=50,
)
```

`precoast=True` turns the quick builder into a two-phase mission. The first
phase coasts from the initial state; the second phase performs the transfer.
`t1_bounds_s` bounds the end of the precoast. `precoast_grid_size` controls how
many candidate departure times are searched before optimization.

Keep `t1_bounds_s` strictly positive for solver-backed runs. A zero-duration
precoast is usually a degenerate phase rather than a useful mission design.

## Pattern 3: Trade Delta-v Against Time

```python
slow = two_burn_rendezvous(x0, xf, w_time=0.0)
fast = two_burn_rendezvous(x0, xf, w_time=2.0)
```

The quick API always minimizes total delta-v. `w_time` adds a final-time term to
the objective. Larger values bias toward shorter transfers, usually at higher
delta-v.

## Pattern 4: Explicit Phase Linking

```python
precoast = Phase(
    name="precoast",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    tof_bounds_s=(0.0, 6_000.0),
    constraints=[constraints.state(x0, where="Front")],
)

transfer = Phase(
    name="transfer",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    previous=precoast,
    link=links.impulsive(),
    tof_bounds_s=(400.0, 60_000.0),
    constraints=[constraints.state(xf, where="Back")],
    variables=[
        variables.ImpulsiveDeltaV(where="Front"),
        variables.ImpulsiveDeltaV(where="Back"),
    ],
)
```

`links.continuous()` means the next phase starts with the same position and
velocity as the previous phase ended.

`links.impulsive()` means position and time remain continuous, but velocity can
jump. Add `ImpulsiveDeltaV(where="Front")` on the linked phase to expose that
jump as a maneuver and objective term.

## Pattern 5: Add Path Constraints

```python
constraints.min_radius(r_min_m, where="Path")
```

Path constraints apply along a phase, not only at boundaries. Minimum radius is
useful for keeping transfer arcs above an altitude floor.

## Pattern 6: Target Orbital Elements

```python
constraints.semi_major_axis(target_a_m, where="Back", tol_m=2.0e3)
constraints.eccentricity(target_e, where="Back", tol=5.0e-3)
constraints.inclination_deg(target_inc_deg, where="Back", tol_deg=0.2)
```

Use orbital-element constraints when the final orbit matters more than a single
Cartesian state. A Cartesian final state can still be supplied as a seed anchor
for the solver.

## Pattern 7: Use J2 Perturbations

```python
dynamics = Dynamics(mu_m3ps2=MU, perturbations=Perturbations(j2=True))
```

J2 is currently implemented in the composable ASSET backend. Other perturbation
flags are reserved for future extensions and fail clearly if requested.

## Pattern 8: Finite Chemical Burns

```python
spacecraft = Spacecraft(
    name="Demo spacecraft",
    dry_mass_kg=500.0,
    thrusters=[
        Thruster(
            name="main",
            thrust_N=2_000.0,
            isp_s=320.0,
            propellant_mass_kg=50.0,
        )
    ],
)
```

Chemical-burn phases use `mode="chemical_burn"` and require a thruster with
positive thrust and specific impulse. The state includes mass, and the solver
tracks propellant usage through mass depletion.

A typical finite-burn transfer uses:

1. departure chemical burn,
2. coast,
3. arrival chemical burn.

Keep the first chemical-burn examples as feasibility solves if the main goal is
to validate structure and report propellant usage rather than optimize fuel.

## Pattern 9: Plot the Result

```python
save_trajectory_html(
    sol.result.traj,
    "trajectory.html",
    maneuvers=sol.result.maneuvers,
    title=mission.name,
)
```

Use maneuver markers for impulsive examples. Use `phase_segments` for
burn-coast-burn examples so the plot shows which part of the trajectory belongs
to each phase.

## Pattern 10: Declare Frame And Scaling

```python
from octavian import Dynamics, SolverScaling
from octavian.coordinates import inertial

dynamics = Dynamics(
    mu_m3ps2=MU,
    frame=inertial("earth", orientation="ECI"),
    scaling=SolverScaling(
        length_m=7_000e3,
        velocity_mps=7_500.0,
        time_s=1_000.0,
        mass_kg=500.0,
    ),
)
```

Explicit scaling is useful when a problem spans very different distance, time,
or mass scales. It does not change the units of mission inputs or solution
arrays. Leave `scaling=None` to use Octavian's endpoint-derived defaults. The
result records both declarations:

```python
print(solution.frame)
print(solution.scaling)
```

## Pattern 11: Change The Central Body

```python
from octavian import SUN, two_burn_rendezvous

mission = two_burn_rendezvous(
    x0,
    xf,
    central_body=SUN,
    tf_bounds_s=(20.0e6, 30.0e6),
)
```

The body declaration supplies gravitational parameter, reference radius, J2
coefficient, and inertial frame origin together. Named body constants override
raw `mu_m3ps2` values so the configuration cannot silently mix Earth and Sun
properties. For a custom object, construct `CelestialBody` explicitly.
