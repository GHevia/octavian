"""User-facing finite-thrust direction and kinematic-attitude configuration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

DirectionFrame = Literal["inertial", "ric"]
ThrustRepresentation = Literal["vector", "fixed", "euler"]


@dataclass(frozen=True, slots=True)
class ThrustControl:
    """Configure how a finite-thrust phase represents its direction.

    ``vector`` preserves Octavian's original three-component vector-throttle
    control while allowing those components to be expressed in inertial or RIC
    axes. ``fixed`` removes direction from the decision vector and optimizes
    only scalar throttle along a prescribed direction. ``euler`` adds
    yaw-pitch-roll kinematic states and angular-rate controls; it does not model
    torque, inertia, or six-degree-of-freedom rotational dynamics.

    Euler angles use a 3-2-1 yaw-pitch-roll convention relative to ``frame``.
    The vehicle body +X axis is the thrust axis. Roll is therefore retained for
    attitude continuity and constraints even though it does not change the
    translational thrust direction.

    Args:
        representation: ``"vector"``, ``"fixed"``, or ``"euler"``.
        frame: Reference axes for the direction or Euler angles. Common
            inertial/ECI and RIC/RTN/LVLH spellings are accepted.
        direction: Fixed direction components; valid only for ``"fixed"``.
        initial_angles_rad: Initial yaw, pitch, and roll for Euler control.
        max_slew_rate_radps: Maximum magnitude of the Euler-angle-rate vector.
        yaw_bounds_rad: Path bounds for yaw.
        pitch_bounds_rad: Path bounds for pitch.
        roll_bounds_rad: Path bounds for roll.
    """

    representation: ThrustRepresentation = "vector"
    frame: DirectionFrame = "inertial"
    direction: tuple[float, float, float] | None = None
    initial_angles_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_slew_rate_radps: float = math.radians(1.0)
    yaw_bounds_rad: tuple[float, float] = (-math.pi, math.pi)
    pitch_bounds_rad: tuple[float, float] = (-0.5 * math.pi, 0.5 * math.pi)
    roll_bounds_rad: tuple[float, float] = (-math.pi, math.pi)

    def __post_init__(self) -> None:
        representation = str(self.representation).strip().lower().replace("-", "_")
        if representation not in {"vector", "fixed", "euler"}:
            raise ValueError("representation must be vector, fixed, or euler")
        frame = _normalize_frame(self.frame)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "frame", frame)

        direction = self.direction
        if representation == "fixed":
            if direction is None:
                raise ValueError("Fixed thrust control requires direction=")
            vector = _unit_vector(direction, name="direction")
            object.__setattr__(self, "direction", tuple(float(value) for value in vector))
        elif direction is not None:
            raise ValueError("direction= is only valid for fixed thrust control")

        angles = _finite_triplet(self.initial_angles_rad, name="initial_angles_rad")
        object.__setattr__(self, "initial_angles_rad", angles)
        rate = float(self.max_slew_rate_radps)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("max_slew_rate_radps must be finite and non-negative")
        object.__setattr__(self, "max_slew_rate_radps", rate)
        for name in ("yaw_bounds_rad", "pitch_bounds_rad", "roll_bounds_rad"):
            bounds = _finite_bounds(getattr(self, name), name=name)
            object.__setattr__(self, name, bounds)
        yaw, pitch, roll = angles
        for name, value, bounds in (
            ("yaw", yaw, self.yaw_bounds_rad),
            ("pitch", pitch, self.pitch_bounds_rad),
            ("roll", roll, self.roll_bounds_rad),
        ):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"Initial {name} angle is outside {name}_bounds_rad")

    @classmethod
    def vector(cls, *, frame: str = "inertial") -> ThrustControl:
        """Create a free three-component vector-throttle control.

        Args:
            frame: Frame in which the optimizer's vector components are
                expressed.

        Returns:
            A vector-throttle configuration.
        """
        return cls(representation="vector", frame=_normalize_frame(frame))

    @classmethod
    def fixed(
        cls,
        direction: Sequence[float],
        *,
        frame: str = "inertial",
    ) -> ThrustControl:
        """Create a prescribed direction with optimized scalar throttle.

        Args:
            direction: Non-zero direction components in ``frame``. The vector
                is normalized automatically.
            frame: Reference frame for ``direction``.

        Returns:
            A fixed-direction scalar-throttle configuration.
        """
        vector = _unit_vector(direction, name="direction")
        return cls(
            representation="fixed",
            frame=_normalize_frame(frame),
            direction=tuple(float(value) for value in vector),
        )

    @classmethod
    def euler(
        cls,
        *,
        frame: str = "inertial",
        initial_angles_rad: Sequence[float] = (0.0, 0.0, 0.0),
        max_slew_rate_radps: float = math.radians(1.0),
        yaw_bounds_rad: Sequence[float] = (-math.pi, math.pi),
        pitch_bounds_rad: Sequence[float] = (-0.5 * math.pi, 0.5 * math.pi),
        roll_bounds_rad: Sequence[float] = (-math.pi, math.pi),
    ) -> ThrustControl:
        """Create yaw-pitch-roll states with bounded angular-rate controls.

        Args:
            frame: Reference frame for the 3-2-1 Euler angles.
            initial_angles_rad: Initial yaw, pitch, and roll in radians.
            max_slew_rate_radps: Maximum Euler-rate-vector magnitude.
            yaw_bounds_rad: Lower and upper yaw path bounds.
            pitch_bounds_rad: Lower and upper pitch path bounds.
            roll_bounds_rad: Lower and upper roll path bounds.

        Returns:
            A kinematic-attitude scalar-throttle configuration.
        """
        return cls(
            representation="euler",
            frame=_normalize_frame(frame),
            initial_angles_rad=_finite_triplet(
                initial_angles_rad,
                name="initial_angles_rad",
            ),
            max_slew_rate_radps=max_slew_rate_radps,
            yaw_bounds_rad=_finite_bounds(yaw_bounds_rad, name="yaw_bounds_rad"),
            pitch_bounds_rad=_finite_bounds(pitch_bounds_rad, name="pitch_bounds_rad"),
            roll_bounds_rad=_finite_bounds(roll_bounds_rad, name="roll_bounds_rad"),
        )

    @property
    def carries_attitude(self) -> bool:
        """Return whether this representation adds Euler attitude states."""
        return self.representation == "euler"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable control metadata for solution reports."""
        return asdict(self)


def euler_thrust_direction(
    angles_rad: Sequence[float],
) -> np.ndarray:
    """Return the body +X thrust direction in the Euler reference frame.

    Args:
        angles_rad: ``[yaw, pitch, roll]`` using the 3-2-1 convention.

    Returns:
        Unit direction components in the configured attitude reference frame.
    """
    yaw, pitch, _roll = _finite_triplet(angles_rad, name="angles_rad")
    return np.asarray(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        ],
        dtype=float,
    )


def _normalize_frame(frame: str) -> DirectionFrame:
    """Normalize common inertial and local-orbital frame aliases."""
    normalized = str(frame).strip().lower().replace("-", "_").replace("/", "_")
    if normalized in {"eci", "j2000", "icrf", "inertial"}:
        return "inertial"
    if normalized in {"ric", "rtn", "lvlh", "ric_rtn_lvlh"}:
        return "ric"
    raise ValueError("frame must be inertial/ECI or RIC/RTN/LVLH")


def _unit_vector(value: Sequence[float], *, name: str) -> np.ndarray:
    """Return a validated unit vector."""
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    return vector / norm


def _finite_triplet(
    value: Sequence[float],
    *,
    name: str,
) -> tuple[float, float, float]:
    """Return a finite three-value tuple."""
    values = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return tuple(float(item) for item in values)  # type: ignore[return-value]


def _finite_bounds(
    value: Sequence[float],
    *,
    name: str,
) -> tuple[float, float]:
    """Return finite ordered lower and upper bounds."""
    lower, upper = np.asarray(value, dtype=float).reshape(2)
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)):
        raise ValueError(f"{name} must contain finite values")
    if not float(lower) < float(upper):
        raise ValueError(f"{name} must satisfy lower < upper")
    return float(lower), float(upper)
