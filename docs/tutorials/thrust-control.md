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
