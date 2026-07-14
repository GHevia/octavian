"""Cartesian inertial and chief-centered LVLH state transformations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..specs import BoundaryState

Matrix3 = NDArray[np.float64]


def lvlh_basis(chief_position_m: ArrayLike, chief_velocity_mps: ArrayLike) -> Matrix3:
    """Return the inertial-to-LVLH direction-cosine matrix.

    Matrix rows are radial, along-track, and orbit-normal unit vectors.
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


def _chief_angular_rate(chief: BoundaryState) -> float:
    radius_sq = float(np.dot(chief.r_m, chief.r_m))
    if radius_sq <= 0.0:
        raise ValueError("Chief position must have non-zero norm")
    return float(np.linalg.norm(np.cross(chief.r_m, chief.v_mps)) / radius_sq)


def inertial_to_relative_state(chief: BoundaryState, deputy: BoundaryState) -> BoundaryState:
    """Transform a deputy inertial state to the chief's instantaneous LVLH frame."""
    dcm = lvlh_basis(chief.r_m, chief.v_mps)
    relative_position = dcm @ (deputy.r_m - chief.r_m)
    inertial_relative_velocity = dcm @ (deputy.v_mps - chief.v_mps)
    omega_lvlh = np.asarray([0.0, 0.0, _chief_angular_rate(chief)], dtype=float)
    relative_velocity = inertial_relative_velocity - np.cross(
        omega_lvlh, relative_position
    )
    return BoundaryState(relative_position, relative_velocity)


def relative_to_inertial_state(chief: BoundaryState, relative: BoundaryState) -> BoundaryState:
    """Transform a chief-centered LVLH state to inertial Cartesian coordinates."""
    dcm = lvlh_basis(chief.r_m, chief.v_mps)
    omega_lvlh = np.asarray([0.0, 0.0, _chief_angular_rate(chief)], dtype=float)
    inertial_position = chief.r_m + dcm.T @ relative.r_m
    inertial_velocity = chief.v_mps + dcm.T @ (
        relative.v_mps + np.cross(omega_lvlh, relative.r_m)
    )
    return BoundaryState(inertial_position, inertial_velocity)
