"""ASSET expressions for finite-thrust control representations."""

from __future__ import annotations

from typing import Any

from ._asset import vf
from .control import ThrustControl


def thrust_vector_and_rate(
    config: ThrustControl,
    *,
    controls: Any,
    position: Any,
    velocity: Any,
    attitude: Any | None = None,
) -> tuple[Any, Any, Any | None]:
    """Return dimensionless inertial thrust, throttle, and physical rates.

    Euler slew controls are normalized to unit magnitude for numerical
    conditioning. This helper scales them by the configured maximum slew rate
    before they enter the attitude kinematics.
    """
    if config.representation == "vector":
        vector = controls
        throttle = vector.norm()
        if config.frame == "ric":
            vector = ric_to_inertial(vector, position, velocity)
        return vector, throttle, None

    throttle = controls[0]
    if config.representation == "fixed":
        direction = vf.stack(
            [position[0] * 0.0 + component for component in (config.direction or (1.0, 0.0, 0.0))]
        )
        if config.frame == "ric":
            direction = ric_to_inertial(direction, position, velocity)
        return throttle * direction, throttle, None

    if attitude is None:
        raise ValueError("Euler thrust control requires attitude states")
    yaw = attitude[0]
    pitch = attitude[1]
    direction = vf.stack(
        [
            vf.cos(pitch) * vf.cos(yaw),
            vf.cos(pitch) * vf.sin(yaw),
            -vf.sin(pitch),
        ]
    )
    if config.frame == "ric":
        direction = ric_to_inertial(direction, position, velocity)
    attitude_rate = float(config.max_slew_rate_radps) * controls.segment(1, 3)
    return throttle * direction, throttle, attitude_rate


def ric_to_inertial(vector_ric: Any, position: Any, velocity: Any) -> Any:
    """Rotate symbolic RIC components into inertial axes."""
    radial = position / position.norm()
    cross_track = position.cross(velocity)
    cross_track = cross_track / cross_track.norm()
    in_track = cross_track.cross(radial)
    return radial * vector_ric[0] + in_track * vector_ric[1] + cross_track * vector_ric[2]
