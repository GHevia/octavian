# Mission Patterns

This tutorial shows how to use the current Octavian options as building blocks.
Start with the quick API when the mission is a standard inertial transfer,
relative hop, or chain of relative transfers. Move to the composable API when
you need explicit phases, links, custom constraints, finite burns, or an
arbitrary burn topology.

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
- custom objectives or per-phase mesh settings,
- chief-centered relative motion, including exact finite burns and coasts.

Use `relative_hop` or `relative_transfer_chain` when the relative mission is:

- a bounded coast followed by a two-impulse hop, or
- multiple two-impulse transfers separated by bounded natural coasts.

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

## Pattern 8: Finite-Thrust Phases

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

Powered phases use `mode="finite_thrust"` and require a thruster with positive
thrust and specific impulse. The chemical-specific compatibility spelling is
`mode="chemical_burn"`. The state includes mass, and the solver tracks
propellant usage through mass depletion.

```python
powered = Phase(
    name="injection",
    mode="finite_thrust",
    spacecraft=spacecraft,
    dynamics=dynamics,
)

mission = Mission(
    phases=[powered],
    objectives=[objectives.minimize_propellant()],
)
```

A common finite-burn transfer uses:

1. departure chemical burn,
2. coast,
3. arrival chemical burn.

This is a mission pattern, not a compiler restriction. Standalone powered
phases and longer powered/coast sequences are supported. Coast phases between
powered phases carry mass automatically; all phases in that chain must use the
same spacecraft configuration.

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

## Pattern 12: Optimize Relative Motion With CWH

```python
dynamics = Dynamics.cwh(
    chief_orbit_radius_m=EARTH.mean_radius_m + 400_000.0,
    central_body=EARTH,
    chief_name="Chief",
    reference_length_m=1_000.0,
)

phase = Phase(
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    initial_state=initial_relative_state,
    final_state=final_relative_state,
    tof_bounds_s=(1_200.0, 2_400.0),
    constraints=[
        constraints.state(initial_relative_state, where="Front"),
        constraints.state(final_relative_state, where="Back"),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)
```

Relative positions and velocities use the chief LVLH/RTN frame. Use
`relative.inertial_to_relative_state(...)` and
`relative.relative_to_inertial_state(...)` to transform analysis states at a
known chief state. CWH assumes a circular chief and small deputy separation;
use a nonlinear model when those assumptions are not appropriate.

For exact central gravity with perturbations, keep the same public RIC states
and use the coupled-ECI formulation:

```python
dynamics = Dynamics.relative(
    chief_initial_state_eci=chief_initial_state_eci,
    central_body=EARTH,
    propagation_mode="coupled_eci",
    perturbations=Perturbations(j2=True, sun=True),
)
```

The compiler uses CWH only to seed the solve. It propagates the chief and deputy
as two absolute states and converts them to RIC for constraints and results.

Compose multiple relative phases with `previous=` exactly as for inertial
missions. The compiler links the formulation's native state, so a coupled ECI
chain preserves both spacecraft states without round-tripping through a
boundary conversion.

When Sun or Moon tables are active, each relative duration contributes to the
cumulative absolute mission-time horizon. The shared table extends through
that absolute upper bound plus `third_body_table_margin_s`; increase the margin
for unusually aggressive time searches or custom solver behavior.

For the common coast–transfer–coast pattern, use the quick builder:

```python
mission = relative_hop(
    initial_ric,
    target_ric,
    chief_initial_state_eci=chief_eci,
    departure_coast_time_bounds_s=(120.0, 600.0),
    transfer_time_bounds_s=(900.0, 1_800.0),
    perturbations=Perturbations(j2=True),
)
```

String complete transfers together by supplying ordered post-arrival states:

```python
mission = relative_transfer_chain(
    initial_ric,
    [inspection_point_ric, final_ric],
    chief_initial_state_eci=chief_eci,
    transfer_time_bounds_s=[(600.0, 1_200.0), (600.0, 1_200.0)],
    coast_time_bounds_s=(300.0, 600.0),
)
```

This example has four impulses. To optimize a three-burn topology, build three
linked composable phases and leave the intermediate burn's velocity free while
constraining only its RIC position waypoint, as shown in composable example 20.

For exact two-body dynamics with native RIC decision variables, select
`propagation_mode="nonlinear_ric"` for a circular chief or `"coupled_ric"` for
a propagated circular/eccentric chief. Then target one component directly:

```python
constraints.ric_state("I", -100.0, where="Back")
```

For a native D'Amico phase, use
`propagation_mode="damico"`, fix the Front vector with
`constraints.relative_orbital_elements(...)`, and target any individual
element with `constraints.relative_orbital_element(...)`. Time remains free
when `tof_bounds_s` is a nonzero interval.

Save state, range, and solar-phase histories beside either trajectory plot:

```python
solution.viz().save_diagnostics_html("relative_diagnostics.html")
```

For analysis-only ROE propagation with the same force-model vocabulary:

```python
from octavian import propagate

history = propagate.relative_elements(
    initial_roe,
    np.linspace(0.0, 6.0 * 3_600.0, 145),
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=Perturbations(j2=True, sun=True),
    initial_epoch="2026-01-01T00:00:00Z",
)

osculating_elements = history.elements
ric_for_plotting = history.ric
```

Use times ending at zero, such as `[-600.0, -300.0, 0.0]`, to generate a
pre-event coast. Use times starting at zero for a post-event coast.

## Pattern 13: Add Relative Safety And Lighting Geometry

```python
constraints.keep_out_sphere(
    radius_m=75.0,
    center_m=[0.0, 0.0, 0.0],
)
constraints.approach_cone(
    axis=[0.0, -1.0, 0.0],
    half_angle_deg=30.0,
)
constraints.lighting_angle(
    sun_direction=[1.0, 0.0, 0.0],
    min_angle_deg=85.0,
    max_angle_deg=121.0,
)
```

All vectors and origins use the phase's declared frame. The approach cone is
one-sided, so the opposite direction does not satisfy it. Lighting bounds use
a fixed direction over the phase; transform or update that direction when a
longer arc needs time-varying Sun geometry. Constraint extrema and satisfaction
flags are included in `solution.result.info["constraint_report"]`.

## Pattern 14: Compose Relative Finite Burns And Coasts

```python
dynamics = Dynamics.relative(
    chief_initial_state_eci=chief_initial_state_eci,
    propagation_mode="coupled_eci",
)

departure = Phase(
    mode="finite_thrust",
    spacecraft=deputy,
    dynamics=dynamics,
    initial_state=initial_ric,
    tof_bounds_s=(50.0, 70.0),
)
coast = Phase(
    mode="relative_coast",
    spacecraft=deputy,
    dynamics=dynamics,
    previous=departure,
    tof_bounds_s=(250.0, 350.0),
    tof_is_relative=True,
)
arrival = Phase(
    mode="finite_thrust",
    spacecraft=deputy,
    dynamics=dynamics,
    previous=coast,
    final_state=target_ric,
    tof_bounds_s=(50.0, 70.0),
    tof_is_relative=True,
)
```

Only the deputy is powered. The chief and deputy both receive their declared
gravity and perturbation accelerations, while the coast keeps deputy mass
constant and continuous between burns. Add
`objectives.minimize_propellant()` to minimize the powered-chain consumption.
Finite thrust currently requires the exact `"coupled_eci"` formulation;
`"nonlinear_ric"`, `"coupled_ric"`, and relative-element modes remain
propulsion-free.

## Pattern 15: Seed A Low-Thrust Spiral

```python
from octavian import guesses

phase = Phase(
    mode="low_thrust",
    spacecraft=electric_spacecraft,
    dynamics=dynamics,
    initial_state=initial_circular_state,
    final_state=terminal_radius_anchor,
    tof_bounds_s=(14 * 3_600.0, 24 * 3_600.0),
    initial_guess=guesses.low_thrust_spiral(
        throttle=0.85,
        direction="auto",
    ),
    constraints=[
        constraints.state(initial_circular_state, where="Front"),
        constraints.semi_major_axis(target_radius_m, where="Back", tol_m=10_000.0),
        constraints.eccentricity(0.01, where="Back", tol=0.0099),
    ],
)
```

The seed estimates tangential spiral delta-v, converts it to burn time through
the rocket equation, and integrates gravity, thrust, and mass at the requested
seed throttle. `direction="auto"` selects prograde for a larger target radius
and retrograde for a smaller one. The optimizer does not retain this steering
law; the vector throttle at every collocation point remains free.

Use a final Cartesian state only as the target-radius and scaling anchor, then
constrain terminal orbital elements to leave longitude free. This built-in seed
assumes a near-circular, approximately coplanar transfer. Adjust `time_scale`
or provide a future custom seed for strongly eccentric or plane-changing arcs.
