# Cislunar CR3BP

Octavian's first cislunar model is the circular restricted three-body problem
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

This first increment supports ballistic phases. Finite thrust, ephemeris
primaries, and transitions between inertial and synodic phases require
additional models and are intentionally rejected rather than silently
approximated.

Osculating two-body orbital-element constraints are not meaningful in this
rotating three-body frame. Target Cartesian synodic states directly.

## Plot The Synodic Geometry

```python
from octavian.viz import save_cr3bp_trajectory_html

save_cr3bp_trajectory_html(
    solution.traj,
    "earth_moon_cr3bp.html",
    system=system,
)
```

The plot includes both primaries and all five Lagrange points. See
`examples/composable/cislunar/22_earth_moon_cr3bp.py` for an executable
propagate–target–solve–convert workflow.
