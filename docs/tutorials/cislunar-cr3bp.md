# Cislunar CR3BP

Octavian's foundational cislunar model is the circular restricted three-body problem
(CR3BP). It represents two massive bodies on circular orbits about their
barycenter and a massless spacecraft moving under their gravity.

The model is useful for early design near Earth–Moon libration points and for
building initial guesses. It is not an ephemeris model: the Earth–Moon
distance and rotation rate are constant, the primaries are point masses, and
the Sun and lunar eccentricity are omitted.

## Define A System

```python
from octavian import CR3BPSystem

system = CR3BPSystem.earth_moon()
```

The default separation is 384,400 km. A general catalog-body system is also
available:

```python
from octavian import EARTH, MOON

system = CR3BPSystem(
    primary=EARTH,
    secondary=MOON,
    separation_m=384_400_000.0,
)
```

The system exposes its canonical scaling and geometry:

```python
print(system.mass_parameter)
print(system.mean_motion_radps)
print(system.period_s)
print(system.lagrange_points(dimensional=False))
```

The synodic frame is barycentric. The primary is fixed at
`[-mu, 0, 0]`, the secondary at `[1-mu, 0, 0]`, and +X points from the
primary toward the secondary.

## Dimensional And Canonical States

Composable phases use SI units: meters, meters per second, and seconds.
Canonical conversion remains explicit:

```python
from octavian.cislunar import (
    dimensionalize_state,
    nondimensionalize_state,
)

canonical = nondimensionalize_state(state_si, system)
state_si_again = dimensionalize_state(canonical, system)
```

Use `nondimensionalize_time` and `dimensionalize_time` for time values. One
canonical distance unit is the primary-secondary separation; one canonical
time unit is the inverse mean motion.

## Synodic And Inertial States

```python
from octavian.cislunar import (
    inertial_to_synodic_state,
    synodic_to_inertial_state,
)

earth_centered_inertial = synodic_to_inertial_state(
    state_synodic,
    time_s=86_400.0,
    system=system,
    origin="earth",
)

recovered = inertial_to_synodic_state(
    earth_centered_inertial,
    time_s=86_400.0,
    system=system,
    origin="primary",
)
```

Supported inertial origins are the barycenter, primary, and secondary. The
optional `phase_at_epoch_rad` rotates the aligned-at-zero convention to a
known inertial phase.

These transforms are CR3BP geometry conversions. They do not query SPICE or
replace high-fidelity ephemeris transformations.

## Propagate And Check Jacobi Conservation

```python
import numpy as np

from octavian import state
from octavian.cislunar import jacobi_constant, propagate_cr3bp

l4 = system.lagrange_points()["L4"]
initial = state(l4 + [100_000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
times_s = np.linspace(0.0, 3.0 * 86_400.0, 301)

history = propagate_cr3bp(
    initial,
    times_s,
    system=system,
    max_step=300.0,
)
constants = [
    jacobi_constant(row[0:6], system=system)
    for row in history
]
```

`propagate_cr3bp` is a deterministic fourth-order Runge–Kutta service for
analysis and initial-state generation. Requested times can increase or
decrease. Tighten `max_step` when long unstable arcs require better invariant
preservation.

## Composable CR3BP Phases

```python
from octavian import Dynamics, Phase

dynamics = Dynamics.cr3bp()

arc = Phase(
    name="synodic_arc",
    mode="coast",
    spacecraft=probe,
    dynamics=dynamics,
    initial_state=initial,
    final_state=target,
    tof_bounds_s=(43_000.0, 44_000.0),
    constraints=[
        constraints.state(initial, where="Front"),
        constraints.state(target, where="Back"),
    ],
)
```

The composable backend uses dimensional CR3BP equations and natural canonical
solver scaling while keeping public inputs and results in SI. Multiple CR3BP
coast phases can be linked when they use the same system. Impulsive links work
through the ordinary composable phase machinery.

CR3BP phases currently support ballistic dynamics and impulsive links. Finite
thrust and ephemeris primaries require a different dynamics model and are
intentionally rejected rather than silently approximated.

Osculating two-body orbital-element constraints are not meaningful in this
rotating three-body frame. Target Cartesian synodic states directly.

## Solve A Periodic Orbit

Periodic-orbit correction uses the same composable constraint vocabulary:

```python
arc = Phase(
    name="L1_planar_Lyapunov",
    mode="coast",
    spacecraft=probe,
    dynamics=Dynamics.cr3bp(),
    initial_state=initial_seed_si,
    final_state=terminal_seed_si,
    tof_bounds_s=(period_min_s, period_max_s),
    constraints=[
        constraints.periodic_state(),
        constraints.state_component("x", x0_m, where="Front"),
        constraints.state_component("y", 0.0, where="Front"),
    ],
)
```

`periodic_state()` creates a direct ASSET front/back equality in the synodic
frame. Time is excluded, so the optimizer remains free to select the period.
The component constraints choose one orbit-family member and a symmetry-plane
crossing. The `initial_state` and `final_state` values seed solver scaling and
the collocation mesh; they are not substitutes for the periodic constraint.

Canonical seeds from CR3BP references remain convenient at the user boundary:
dimensionalize them before declaring the phase, then nondimensionalize the
solved rows for family comparisons and canonical plotting.

To select the family member by invariant instead of initial x, use:

```python
constraints=[
    constraints.periodic_state(),
    constraints.state_component("y", 0.0, where="Front"),
    constraints.jacobi_constant(3.16, dimensional=False),
]
```

The Jacobi target is applied directly to the synodic phase state. Canonical
targets use `dimensional=False`; the default accepts dimensional `m²/s²`.
`where` may be `"Front"`, `"Back"`, or `"Path"`, and an optional `tolerance`
creates symmetric bounds instead of an equality.

## Transfer And Increase Fidelity

Compatible CR3BP coasts can be joined with ordinary continuous or impulsive
links. This supports a direct sequence such as an L1 orbit coast, impulsive
departure, free-time transfer, L2 insertion, and L2 orbit coast.

To assess a design under J2, ephemeris Sun/Moon gravity, drag, or SRP, first
convert selected synodic states to an inertial frame and solve a separate
inertial mission with `Dynamics.for_body(...)`. Aligning the circular
synodic geometry with the BSP Moon at the handoff epoch keeps the initial
geometry consistent. The resulting arc is not a CR3BP periodic orbit; it is a
perturbed-model retargeting problem with explicit boundary corrections.

## Plot The Synodic Geometry

```python
from octavian.viz import save_cr3bp_trajectory_html

save_cr3bp_trajectory_html(
    solution.traj,
    "earth_moon_cr3bp.html",
    system=system,
)
```

The plot includes both primaries and all five Lagrange points. It can also
overlay reference periodic orbits, color phase segments, and mark maneuvers.

Run the executable progression:

- `examples/composable/cislunar/22_earth_moon_cr3bp.py` — dimensional
  propagate–target–solve–convert workflow;
- `examples/composable/cislunar/24_canonical_periodic_orbit.py` — canonical
  L1 periodic-orbit correction;
- `examples/composable/cislunar/25_periodic_orbit_transfer.py` — L1-to-L2
  coast/impulse/transfer/impulse/coast mission;
- `examples/composable/cislunar/26_high_fidelity_recapture.py` — BSP-aligned
  handoff to inertial J2, Sun/Moon, and SRP dynamics.
- `examples/composable/cislunar/27_jacobi_targeted_periodic_orbit.py` —
  periodic-orbit correction with a canonical Jacobi family target.

The [cislunar example guide](../examples/cislunar.md) explains the design
choices and current fidelity boundaries in detail.
