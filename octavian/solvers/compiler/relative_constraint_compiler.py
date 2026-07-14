"""Compilation and reporting for Cartesian relative-geometry constraints."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._asset import vf
from ...constraints import ApproachCone, Constraint, KeepOutSphere, LightingAngle
from ...phase import Phase

RelativeGeometryConstraint = KeepOutSphere | ApproachCone | LightingAngle


def is_relative_geometry_constraint(constraint: Constraint) -> bool:
    """Return whether this module compiles the supplied constraint."""
    return isinstance(constraint, (KeepOutSphere, ApproachCone, LightingAngle))


def relative_geometry_constraints(phase: Phase) -> tuple[RelativeGeometryConstraint, ...]:
    """Return the phase constraints owned by this compiler."""
    return tuple(
        constraint
        for constraint in phase.constraints
        if is_relative_geometry_constraint(constraint)
    )


def apply_relative_geometry_constraint(
    asset_phase: Any,
    constraint: RelativeGeometryConstraint,
    position_indices: tuple[int, ...] = (0, 1, 2),
) -> None:
    """Compile one keep-out, approach-cone, or lighting-angle inequality."""
    position = vf.Arguments(3)
    where = constraint.where

    if isinstance(constraint, KeepOutSphere):
        offset = position - np.asarray(constraint.center_m, dtype=float)
        signed_violation = constraint.radius_m**2 - offset.dot(offset)
        asset_phase.addInequalCon(
            where,
            vf.stack([signed_violation]),
            list(position_indices),
        )
        return

    if isinstance(constraint, ApproachCone):
        offset = position - np.asarray(constraint.vertex_m, dtype=float)
        axial_distance = offset.dot(np.asarray(constraint.axis, dtype=float))
        cosine_sq = float(np.cos(np.deg2rad(constraint.half_angle_deg)) ** 2)
        cone_violation = cosine_sq * offset.dot(offset) - axial_distance**2
        asset_phase.addInequalCon(
            where,
            vf.stack([cone_violation]),
            list(position_indices),
        )
        # Squaring the cone inequality admits the opposite cone. This second
        # inequality retains only the forward half selected by ``axis``.
        asset_phase.addInequalCon(
            where,
            vf.stack([-axial_distance]),
            list(position_indices),
        )
        return

    if isinstance(constraint, LightingAngle):
        offset = position - np.asarray(constraint.origin_m, dtype=float)
        distance = offset.norm()
        projection = offset.dot(np.asarray(constraint.sun_direction, dtype=float))
        min_cosine = float(np.cos(np.deg2rad(constraint.min_angle_deg)))
        max_cosine = float(np.cos(np.deg2rad(constraint.max_angle_deg)))
        asset_phase.addInequalCon(
            where,
            vf.stack([max_cosine * distance - projection]),
            list(position_indices),
        )
        asset_phase.addInequalCon(
            where,
            vf.stack([projection - min_cosine * distance]),
            list(position_indices),
        )
        return

    raise TypeError(f"Unsupported relative geometry constraint {type(constraint).__name__}")


def relative_constraint_report_rows(
    *,
    phase_name: str,
    constraint: RelativeGeometryConstraint,
    phase_traj: np.ndarray,
) -> list[dict[str, float | str | bool]]:
    """Evaluate one relative-geometry constraint against a solved trajectory."""
    positions = _positions_at_location(phase_traj, constraint.where)

    if isinstance(constraint, KeepOutSphere):
        distances = np.linalg.norm(positions - constraint.center_m, axis=1)
        actual = float(np.min(distances))
        target = float(constraint.radius_m)
        tolerance = max(1e-6, 1e-7 * target)
        return [
            _report_row(
                phase_name,
                constraint,
                name="keep_out_sphere",
                target=target,
                actual=actual,
                error=actual - target,
                tolerance=tolerance,
                satisfied=actual >= target - tolerance,
            )
        ]

    if isinstance(constraint, ApproachCone):
        offsets = positions - constraint.vertex_m
        angles = _angles_deg(offsets, np.asarray(constraint.axis, dtype=float))
        actual = float(np.max(angles)) if angles.size else float("nan")
        target = float(constraint.half_angle_deg)
        tolerance = 1e-4
        return [
            _report_row(
                phase_name,
                constraint,
                name="approach_cone",
                target=target,
                actual=actual,
                error=actual - target,
                tolerance=tolerance,
                satisfied=bool(np.isfinite(actual) and actual <= target + tolerance),
            )
        ]

    if isinstance(constraint, LightingAngle):
        offsets = positions - constraint.origin_m
        angles = _angles_deg(
            offsets,
            np.asarray(constraint.sun_direction, dtype=float),
        )
        if not angles.size:
            minimum_actual = maximum_actual = float("nan")
        else:
            minimum_actual = float(np.min(angles))
            maximum_actual = float(np.max(angles))
        tolerance = 1e-4
        return [
            _report_row(
                phase_name,
                constraint,
                name="lighting_min_angle_deg",
                target=float(constraint.min_angle_deg),
                actual=minimum_actual,
                error=minimum_actual - float(constraint.min_angle_deg),
                tolerance=tolerance,
                satisfied=bool(
                    np.isfinite(minimum_actual)
                    and minimum_actual >= constraint.min_angle_deg - tolerance
                ),
            ),
            _report_row(
                phase_name,
                constraint,
                name="lighting_max_angle_deg",
                target=float(constraint.max_angle_deg),
                actual=maximum_actual,
                error=maximum_actual - float(constraint.max_angle_deg),
                tolerance=tolerance,
                satisfied=bool(
                    np.isfinite(maximum_actual)
                    and maximum_actual <= constraint.max_angle_deg + tolerance
                ),
            ),
        ]

    raise TypeError(f"Unsupported relative geometry constraint {type(constraint).__name__}")


def _positions_at_location(phase_traj: np.ndarray, where: str) -> np.ndarray:
    positions = np.asarray(phase_traj, dtype=float)[:, 0:3]
    if where == "Front":
        return positions[0:1]
    if where == "Back":
        return positions[-1:]
    return positions


def _angles_deg(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-12
    if not np.any(valid):
        return np.empty(0, dtype=float)
    cosines = (vectors[valid] @ direction) / norms[valid]
    return np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))


def _report_row(
    phase_name: str,
    constraint: Constraint,
    *,
    name: str,
    target: float,
    actual: float,
    error: float,
    tolerance: float,
    satisfied: bool,
) -> dict[str, float | str | bool]:
    return {
        "phase": phase_name,
        "where": constraint.where,
        "family": constraint.family,
        "constraint": name,
        "target": target,
        "tolerance": tolerance,
        "actual": actual,
        "error": error,
        "satisfied": satisfied,
    }
