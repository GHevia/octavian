"""CR3BP dimensional scaling and synodic/inertial transformations."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..specs import BoundaryState
from .model import CR3BPSystem


def nondimensionalize_state(
    state_synodic: BoundaryState,
    system: CR3BPSystem,
) -> BoundaryState:
    """Convert a dimensional synodic state to CR3BP canonical units."""
    return BoundaryState(
        state_synodic.r_m / system.separation_m,
        state_synodic.v_mps / system.velocity_scale_mps,
    )


def dimensionalize_state(
    state_nondimensional: BoundaryState,
    system: CR3BPSystem,
) -> BoundaryState:
    """Convert a canonical CR3BP state to dimensional synodic SI units."""
    return BoundaryState(
        state_nondimensional.r_m * system.separation_m,
        state_nondimensional.v_mps * system.velocity_scale_mps,
    )


def nondimensionalize_time(time_s: float | Sequence[float], system: CR3BPSystem):
    """Convert seconds to canonical CR3BP time."""
    result = np.asarray(time_s, dtype=float) / system.time_scale_s
    return float(result) if result.ndim == 0 else result


def dimensionalize_time(time_nondimensional: float | Sequence[float], system: CR3BPSystem):
    """Convert canonical CR3BP time to seconds."""
    result = np.asarray(time_nondimensional, dtype=float) * system.time_scale_s
    return float(result) if result.ndim == 0 else result


def synodic_to_inertial_state(
    state_synodic: BoundaryState,
    *,
    time_s: float,
    system: CR3BPSystem,
    origin: str = "primary",
    phase_at_epoch_rad: float = 0.0,
) -> BoundaryState:
    """Rotate a dimensional barycentric synodic state into inertial axes.

    Args:
        state_synodic: Dimensional barycentric synodic state.
        time_s: Seconds from the frame alignment epoch.
        system: CR3BP system.
        origin: ``"barycenter"``, ``"primary"``, or ``"secondary"``.
        phase_at_epoch_rad: Inertial angle of synodic +X at time zero.

    Returns:
        State in inertial axes relative to the selected origin.
    """
    _validate_rotation_inputs(time_s, phase_at_epoch_rad)
    origin_position = _origin_position_synodic(system, origin)
    relative_position = state_synodic.r_m - origin_position
    rotation = _rotation_z(float(phase_at_epoch_rad) + system.mean_motion_radps * float(time_s))
    omega_cross_r = np.cross(
        [0.0, 0.0, system.mean_motion_radps],
        relative_position,
    )
    return BoundaryState(
        rotation @ relative_position,
        rotation @ (state_synodic.v_mps + omega_cross_r),
    )


def inertial_to_synodic_state(
    state_inertial: BoundaryState,
    *,
    time_s: float,
    system: CR3BPSystem,
    origin: str = "primary",
    phase_at_epoch_rad: float = 0.0,
) -> BoundaryState:
    """Rotate an origin-centered inertial state into barycentric synodic axes."""
    _validate_rotation_inputs(time_s, phase_at_epoch_rad)
    rotation = _rotation_z(-(float(phase_at_epoch_rad) + system.mean_motion_radps * float(time_s)))
    relative_position = rotation @ state_inertial.r_m
    rotating_velocity = rotation @ state_inertial.v_mps - np.cross(
        [0.0, 0.0, system.mean_motion_radps],
        relative_position,
    )
    return BoundaryState(
        relative_position + _origin_position_synodic(system, origin),
        rotating_velocity,
    )


def _origin_position_synodic(system: CR3BPSystem, origin: str) -> np.ndarray:
    """Return a supported origin's fixed dimensional synodic position."""
    normalized = str(origin).strip().lower().replace("-", "_")
    if normalized in {"barycenter", "barycentre", "ssb"}:
        return np.zeros(3, dtype=float)
    if normalized in {"primary", system.primary.name}:  # type: ignore[union-attr]
        return system.primary_position_m
    if normalized in {"secondary", system.secondary.name}:  # type: ignore[union-attr]
        return system.secondary_position_m
    raise ValueError("origin must be barycenter, primary, or secondary")


def _rotation_z(angle_rad: float) -> np.ndarray:
    """Return a right-handed passive-to-active Z rotation matrix."""
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _validate_rotation_inputs(time_s: float, phase_at_epoch_rad: float) -> None:
    """Validate scalar frame-rotation inputs."""
    if not math.isfinite(float(time_s)):
        raise ValueError("time_s must be finite")
    if not math.isfinite(float(phase_at_epoch_rad)):
        raise ValueError("phase_at_epoch_rad must be finite")
