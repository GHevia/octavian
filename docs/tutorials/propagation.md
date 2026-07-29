# Analysis Propagation

Optimization missions are built with `Mission` and `Phase`. For propagation
and initial-condition studies that do not need a solve, start with the
`octavian.propagate` namespace:

```python
from octavian import propagate
```

It collects the package's analysis propagators without hiding which physical
model is selected.

## Available Models

| Call | Model | Return |
| --- | --- | --- |
| `propagate.two_body(...)` | Elliptic point-mass inertial motion | `[r, v, time]` array |
| `propagate.cwh(...)` | Linear circular-chief CWH | `[RIC state, time]` array |
| `propagate.nonlinear_ric(...)` | Exact circular-chief RIC before linearization | `[RIC state, time]` array |
| `propagate.relative(...)` | Coupled chief/deputy absolute states | `RelativePropagationResult` |
| `propagate.relative_elements(...)` | D'Amico or classical relative elements | `RelativeElementPropagationResult` |
| `propagate.cr3bp(...)` | Dimensional or canonical synodic CR3BP | `[state, time]` array |

Every ordinary state-history array has seven columns, with time last. The
relative result objects keep multiple useful views together instead of
discarding information.

## Inertial And Reduced Relative Histories

```python
import numpy as np
from octavian import CR3BPSystem, EARTH, Perturbations, propagate, state

radius_m = EARTH.mean_radius_m + 500_000.0
chief = state(
    [radius_m, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
)
times_s = np.linspace(0.0, 600.0, 13)

inertial = propagate.two_body(
    chief,
    times_s,
    mu_m3ps2=EARTH.mu_m3ps2,
)

mean_motion = np.sqrt(EARTH.mu_m3ps2 / radius_m**3)
relative = propagate.cwh(
    [100.0, -500.0, 0.0, 0.0, 0.02, 0.0],
    times_s,
    mean_motion_radps=mean_motion,
)
```

CWH is deliberately named in the call. Use `nonlinear_ric` when the full
circular-chief relative equations are needed, or `relative` to propagate an
arbitrary chief and deputy as absolute Cartesian states.

## Coupled Relative Propagation

```python
relative_initial = state(
    [100.0, -500.0, 0.0],
    [0.0, 0.02, 0.0],
)
result = propagate.relative(
    chief,
    relative_initial,
    times_s,
    perturbations=Perturbations(j2=True),
)

ric = result.relative_trajectory_ric
chief_eci = result.chief_trajectory_eci
deputy_eci = result.deputy_trajectory_eci
```

The same call accepts Moon/Sun gravity, drag, and SRP. Supply
`chief_spacecraft` and `deputy_spacecraft` for differential cannonball forces,
and `initial_epoch` whenever an ephemeris is required.

## Relative Elements Without Duplicate Propagation

`relative_elements` returns both representations from one coupled numerical
history:

```python
result = propagate.relative_elements(
    initial_roe,
    times_s,
    chief_initial_state_eci=chief,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=Perturbations(j2=True, drag=True),
    chief_spacecraft=chief_vehicle,
    deputy_spacecraft=deputy_vehicle,
)

osculating_elements = result.elements
ric_for_plotting = result.ric
```

`result.representation` records whether the native rows are D'Amico or
classical relative elements, and `result.times_s` returns their shared time
vector.

## CR3BP

```python
system = CR3BPSystem.earth_moon()
l4 = state(
    system.lagrange_points(dimensional=False)["L4"],
    [0.0, 0.0, 0.0],
)
history = propagate.cr3bp(
    l4,
    [0.0, 0.01],
    system=system,
    dimensional=False,
)
```

The `dimensional` flag applies to both state and time. See the Cislunar CR3BP
tutorial for conversions, Jacobi diagnostics, and composable optimization.

## Specialized Functions Remain Available

The umbrella namespace is additive. Existing imports such as
`octavian.relative.propagate_cwh`,
`octavian.relative.propagate_relative_numerical`, and
`octavian.cislunar.propagate_cr3bp` remain supported for code that benefits
from the domain-specific module layout.

The complete executable overview is
`examples/analysis/01_propagation_namespace.py`.
