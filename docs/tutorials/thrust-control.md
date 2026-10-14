# Thrust Direction And Kinematic Attitude

Finite-thrust phases use `Phase.thrust_control` to separate propulsion from
the choice of direction variables. The translational force and propellant
model remain unchanged.

## Free Vector-Throttle Control

The default is a dimensionless three-component vector. Its norm is throttle
and is bounded by one; its direction is the thrust direction.

```python
from octavian import ThrustControl

inertial_control = ThrustControl.vector(frame="inertial")
ric_control = ThrustControl.vector(frame="ric")
```

An RIC vector is rotated into inertial coordinates from the propagated state
at every collocation point. For an inertial mission, RIC is defined by that
spacecraft's position and velocity. For exact relative motion, it is defined
by the propagated chief.

Use this representation when the optimizer should choose both throttle and
direction freely.

## Prescribed Direction

Fixed control removes direction from the decision vector and leaves only
scalar throttle:

```python
constant_inertial = ThrustControl.fixed(
    [1.0, 0.0, 0.0],
    frame="inertial",
)

always_in_track = ThrustControl.fixed(
    [0.0, 1.0, 0.0],
    frame="ric",
)
```

Input vectors are normalized. A fixed inertial direction remains constant in
inertial space. A fixed RIC direction follows the local orbital frame.

## Euler-Angle Kinematics

Euler control adds yaw, pitch, and roll states. It uses a 3-2-1 convention
relative to the selected frame, and the body +X axis is the thrust direction.
The controls are scalar throttle plus yaw, pitch, and roll rates.

```python
import numpy as np

from octavian import ThrustControl

attitude = ThrustControl.euler(
    frame="ric",
    initial_angles_rad=np.deg2rad([90.0, 0.0, 0.0]),
    max_slew_rate_radps=np.deg2rad(0.5),
    yaw_bounds_rad=np.deg2rad([-180.0, 180.0]),
    pitch_bounds_rad=np.deg2rad([-80.0, 80.0]),
    roll_bounds_rad=np.deg2rad([-180.0, 180.0]),
)
```

This is a kinematic attitude model. It does not introduce torque, inertia,
reaction wheels, or full six-degree-of-freedom dynamics. Roll is carried so
attitude can remain continuous even though rotation about body +X does not
change the translational thrust direction.

The maximum slew rate bounds the magnitude of the Euler-rate vector, not each
axis independently. Internally, Octavian normalizes the optimizer variables
for conditioning; returned histories are always in physical radians per
second.

## Attach Control To A Phase

```python
from octavian import Phase

burn = Phase(
    name="departure_burn",
    mode="finite_thrust",
    spacecraft=spacecraft,
    dynamics=dynamics,
    thrust_control=attitude,
    # states, time bounds, and constraints...
)
```

The same declarations work for ordinary inertial dynamics and exact
`Dynamics.relative(..., propagation_mode="coupled_eci")` deputy burns.

## Continuous Burn-Coast-Burn Attitude

In a powered chain, a kinematic attitude can continue through an intermediate
coast:

```python
departure = Phase(
    name="departure",
    mode="finite_thrust",
    spacecraft=spacecraft,
    dynamics=dynamics,
    thrust_control=attitude,
)

coast = Phase(
    name="coast",
    mode="coast",
    previous=departure,
)

arrival = Phase(
    name="arrival",
    mode="finite_thrust",
    previous=coast,
)
```

The later phases inherit `thrust_control`. Octavian carries mass and attitude
through the coast and links yaw, pitch, and roll continuously at both
boundaries. Every phase from the first through the last powered phase must use
the same Euler reference frame. Set `thrust_control` explicitly on a later
phase only when changing another compatible setting such as its angle bounds.

## Read The Result

```python
solution = mission.solve()

for control_history in solution.phase_control_trajectories:
    # [time, controls...]
    print(control_history.shape)

for attitude_history in solution.attitude_phase_trajectories:
    # [yaw, pitch, roll, time, yaw_rate, pitch_rate, roll_rate]
    print(attitude_history[-1])
```

For powered Euler phases, `phase_control_trajectories` contains
`[time, throttle, yaw_rate, pitch_rate, roll_rate]`. Euler coasts contain
`[time, yaw_rate, pitch_rate, roll_rate]`. Rates are radians per second.
`solution.result.info["thrust_controls"]` records the declaration used by each
powered phase for reproducible output.

See
`examples/composable/earth_centered/12_thrust_frames_and_attitude.py`
for a complete executable mission.

## Low-thrust Q-law guesses and firing plots

Use a Q-law guess when an inertial low-thrust transfer changes the orbit's
size, eccentricity vector, or plane:

```python
initial_guess = guesses.low_thrust_qlaw(throttle=0.85, plane_tol_deg=0.05)
```

Attach it as `Phase.initial_guess` and provide `Phase.final_state` on the
**desired orbit**. That state's position and velocity define the complete seed
target, including the orbital plane; its orbital phase is not targeted.
Declare terminal constraints separately, for example semi-major axis,
eccentricity, and inclination. Constraints may leave the ascending node free
even though the seed steers toward the anchor's particular plane.

This is a numerical equinoctial proximity-quotient variant of Q-law: squared
errors in `[p, f, g, h, k]` are normalized by their maximum Gauss element rates
per unit acceleration, sampled around the current orbit. Thrust minimizes the
instantaneous derivative of that quotient. The analytic gradient includes
changes in the rate normalization and averages subgradients at tied maxima,
which avoids selecting an arbitrary thrust bias at circularity. Circular and equatorial orbits are supported;
retrograde-equatorial and unbound orbits are rejected. This variant has no
periapsis penalty or efficiency-based coasting. See the
[NASA comparison of Q-law guidance](https://ntrs.nasa.gov/api/citations/20140011269/downloads/20140011269.pdf)
for the original classical-element formulation.

The default seed integrates osculating equinoctial elements, avoiding secular
Cartesian energy drift on long arcs. Cartesian positions and velocities are
resampled together with Hermite interpolation rather than cutting across
orbital arcs with straight lines. Step sizes decrease near the target to
avoid repeatedly overshooting tight arrival tolerances.

For many-revolution, small-acceleration transfers, explicitly choose an
orbit-averaged guess:

```python
initial_guess = guesses.low_thrust_qlaw(averaged=True, orbit_samples=64)
```

This time-averages the Gauss rates using Keplerian residence-time weights and
propagates mean longitude, with at most a quarter-orbit step. It reconstructs
Cartesian samples by solving Kepler's equation. `orbit_samples` controls both
the maximum-rate sampling and the averaging quadrature; `steps_per_orbit`
controls only the default osculating integration. Averaging neglects short-period
oscillations and produces an approximate seed, so use the osculating mode when
those oscillations matter. The subsequent direct solver still uses the full
phase dynamics and needs sufficient mesh resolution for every revolution;
fast seed generation does not remove that optimization cost.
`seed_averaged`, `seed_control_evaluations`, and `seed_estimated_revolutions`
identify the computation performed in the guess metadata.

The seed propagates **two-body** dynamics with constant thrust and Isp. It is
currently available for a single inertial `low_thrust` phase using vector
controls in either the inertial or RIC frame. The optimized phase still uses
its declared force model and constraints. Seed tolerances determine when
steering ends, not solver accuracy. If arrival precedes the minimum duration,
the seed coasts to that minimum. If the maximum duration is reached first, the
partial seed is returned with `seed_target_reached=False`. When compiling a
vector-control seed, coast nodes receive a `1e-4` throttle warm-start vector
because the mass-flow norm is nondifferentiable at zero. The floor is recorded
as `seed_optimizer_throttle_floor`; it does not constrain optimized throttle
and is not applied to the standalone propagated seed. Propellant exhaustion
raises an error. Inspect `result.info["phase_guess_info"]` for arrival errors.

To show actual throttle or firing intervals, including coasts **inside** a
powered phase:

```python
solution.viz().save_html("throttle.html", color_by="throttle")
solution.viz().save_html("firing.html", color_by="firing", firing_threshold=0.001)
```

The default `color_by="phase"` preserves phase colors. Throttle plots use a
fixed 0–1 colorbar. Firing plots use orange above the threshold and gray at or
below it, with boundaries interpolated from sampled throttle. Hover labels
show physical time, throttle, and thrust in newtons. Histories are handled
separately by phase so thrust discontinuities at links are preserved.

Both inertial and relative Plotly views support these options. Lower-level
plotters accept `thrust_histories=solution.phase_thrust_trajectories`; each
history contains `[time_s, throttle, thrust_N]`. Vector norms and scalar
throttles are converted by the solver, so Euler slew rates are never mistaken
for throttle. Explicit coast phases have zero thrust. These plots show the
sampled/interpolated solved control, not an independently validated switching
schedule.

See `examples/composable/earth_centered/11b_low_thrust_plane_change.py` for a
100 mN, 23 kg small-satellite orbit raise from 7,000 to 7,200 km with a
two-degree inclination change, reference orbits, and both throttle and firing
views. It uses a scheduled 32–32.2 hour arrival window and 120 fixed LGL5 segments for a
quick tutorial. Select the collocation order explicitly with
`SolverOptions(transcription_mode="LGL5")`; LGL3 remains the default and LGL7
is also available. The configuration-file key is `transcription_mode`. Broad free-time searches, larger changes, and lower
thrust-to-mass ratios require longer trajectories and larger direct solves.
An ASSET feasibility recovery can return a constraint-satisfying trajectory
without establishing optimality; inspect the optimizer output when judging
fuel optimality.
