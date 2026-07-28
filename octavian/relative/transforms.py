"""Cartesian inertial and chief-centered RIC state transformations.

Octavian uses the common RIC/RTN/LVLH convention: radial, in-track, and
cross-track axes.  State transforms include the rotating-frame velocity term;
simply rotating an inertial velocity difference is not a valid relative
velocity transformation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..specs import BoundaryState

Matrix3 = NDArray[np.float64]
StateHistory = NDArray[np.float64]


def ric_basis(chief_position_m: ArrayLike, chief_velocity_mps: ArrayLike) -> Matrix3:
    """Return the inertial-to-RIC direction-cosine matrix.

    Matrix rows are radial, in-track, and cross-track unit vectors.  Left
    multiplication therefore rotates an inertial vector into RIC coordinates;
    the transpose rotates a RIC vector back to inertial coordinates.
    """
    position = np.asarray(chief_position_m, dtype=float).reshape(3)
    velocity = np.asarray(chief_velocity_mps, dtype=float).reshape(3)
    radius = float(np.linalg.norm(position))
    angular_momentum = np.cross(position, velocity)
    momentum_norm = float(np.linalg.norm(angular_momentum))
    if radius <= 0.0:
        raise ValueError("chief_position_m must have non-zero norm")
    if momentum_norm <= 0.0:
        raise ValueError("Chief position and velocity must define an orbital plane")
    radial = position / radius
    normal = angular_momentum / momentum_norm
    along_track = np.cross(normal, radial)
    return np.vstack([radial, along_track, normal])


def lvlh_basis(chief_position_m: ArrayLike, chief_velocity_mps: ArrayLike) -> Matrix3:
    """Return the inertial-to-LVLH/RTN matrix.

    This compatibility name is exactly equivalent to :func:`ric_basis`.
    """
    return ric_basis(chief_position_m, chief_velocity_mps)


def chief_ric_angular_velocity(chief: BoundaryState) -> NDArray[np.float64]:
    """Return the chief RIC frame angular velocity expressed in RIC.

    The result is ``[0, 0, h/r²]``.  It is exact for the instantaneous RIC
    frame when the orbit plane is fixed, including eccentric chief orbits.
    """
    radius_sq = float(np.dot(chief.r_m, chief.r_m))
    if radius_sq <= 0.0:
        raise ValueError("Chief position must have non-zero norm")
    angular_momentum = float(np.linalg.norm(np.cross(chief.r_m, chief.v_mps)))
    if angular_momentum <= 0.0:
        raise ValueError("Chief position and velocity must define an orbital plane")
    return np.asarray([0.0, 0.0, angular_momentum / radius_sq], dtype=float)


def inertial_to_relative_state(chief: BoundaryState, deputy: BoundaryState) -> BoundaryState:
    """Transform a deputy inertial state to the chief's instantaneous RIC frame."""
    dcm = ric_basis(chief.r_m, chief.v_mps)
    relative_position = dcm @ (deputy.r_m - chief.r_m)
    inertial_relative_velocity = dcm @ (deputy.v_mps - chief.v_mps)
    relative_velocity = inertial_relative_velocity - np.cross(
        chief_ric_angular_velocity(chief), relative_position
    )
    return BoundaryState(relative_position, relative_velocity)


def relative_to_inertial_state(chief: BoundaryState, relative: BoundaryState) -> BoundaryState:
    """Transform a chief-centered RIC state to inertial Cartesian coordinates."""
    dcm = ric_basis(chief.r_m, chief.v_mps)
    inertial_position = chief.r_m + dcm.T @ relative.r_m
    inertial_velocity = chief.v_mps + dcm.T @ (
        relative.v_mps + np.cross(chief_ric_angular_velocity(chief), relative.r_m)
    )
    return BoundaryState(inertial_position, inertial_velocity)


def absolute_to_relative_state(
    chief: BoundaryState,
    deputy: BoundaryState,
) -> BoundaryState:
    """Alias for :func:`inertial_to_relative_state` with explicit terminology."""
    return inertial_to_relative_state(chief, deputy)


def relative_to_absolute_state(
    chief: BoundaryState,
    relative: BoundaryState,
) -> BoundaryState:
    """Return the reconstructed deputy absolute Cartesian state."""
    return relative_to_inertial_state(chief, relative)


def absolute_to_relative_history(
    chief_history: ArrayLike,
    deputy_history: ArrayLike,
) -> StateHistory:
    """Convert matching absolute histories to ``[rho_RIC, rho_dot_RIC, t]``.

    Input rows may contain either six state columns or seven columns with time.
    If time is present in both histories, values must match and are preserved.
    """
    chief_rows, deputy_rows, include_time = _matching_histories(
        chief_history, deputy_history
    )
    converted = np.empty((chief_rows.shape[0], 7 if include_time else 6), dtype=float)
    for index, (chief_row, deputy_row) in enumerate(
        zip(chief_rows, deputy_rows, strict=True)
    ):
        relative = inertial_to_relative_state(
            BoundaryState(chief_row[0:3], chief_row[3:6]),
            BoundaryState(deputy_row[0:3], deputy_row[3:6]),
        )
        converted[index, 0:6] = np.hstack([relative.r_m, relative.v_mps])
    if include_time:
        converted[:, 6] = chief_rows[:, 6]
    return converted


def relative_to_absolute_history(
    chief_history: ArrayLike,
    relative_history: ArrayLike,
) -> StateHistory:
    """Reconstruct deputy absolute history from matching chief and RIC rows.

    Input rows may contain either six state columns or seven columns with time.
    If time is present in both histories, values must match and are preserved.
    """
    chief_rows, relative_rows, include_time = _matching_histories(
        chief_history, relative_history
    )
    converted = np.empty((chief_rows.shape[0], 7 if include_time else 6), dtype=float)
    for index, (chief_row, relative_row) in enumerate(
        zip(chief_rows, relative_rows, strict=True)
    ):
        deputy = relative_to_inertial_state(
            BoundaryState(chief_row[0:3], chief_row[3:6]),
            BoundaryState(relative_row[0:3], relative_row[3:6]),
        )
        converted[index, 0:6] = np.hstack([deputy.r_m, deputy.v_mps])
    if include_time:
        converted[:, 6] = chief_rows[:, 6]
    return converted


def _matching_histories(
    first: ArrayLike,
    second: ArrayLike,
) -> tuple[StateHistory, StateHistory, bool]:
    first_rows = np.asarray(first, dtype=float)
    second_rows = np.asarray(second, dtype=float)
    if first_rows.ndim != 2 or second_rows.ndim != 2:
        raise ValueError("State histories must be two-dimensional arrays")
    if first_rows.shape[0] != second_rows.shape[0]:
        raise ValueError("State histories must contain the same number of rows")
    if first_rows.shape[1] not in (6, 7) or second_rows.shape[1] not in (6, 7):
        raise ValueError("State histories must have six state columns and optional time")
    if not np.all(np.isfinite(first_rows)) or not np.all(np.isfinite(second_rows)):
        raise ValueError("State histories must contain finite values")
    include_time = first_rows.shape[1] == 7 or second_rows.shape[1] == 7
    if include_time and first_rows.shape[1] != second_rows.shape[1]:
        raise ValueError("Both state histories must include time when either one does")
    if include_time and not np.allclose(first_rows[:, 6], second_rows[:, 6]):
        raise ValueError("State-history time columns must match")
    return first_rows, second_rows, include_time
