"""Compatibility helpers for Cartesian relative-geometry constraints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..._asset import vf
from ...constraints import (
    ApproachCone,
    Constraint,
    KeepOutSphere,
    LightingAngle,
    SolarPhaseAngle,
)
from ...phase import Phase
from ..constraint_compiler import ConstraintContext, ConstraintReport

RelativeGeometryConstraint = (
    KeepOutSphere | ApproachCone | LightingAngle | SolarPhaseAngle
)


def is_relative_geometry_constraint(constraint: Constraint) -> bool:
    """Return whether the supplied constraint targets relative geometry."""
    return constraint.family == "relative_geometry"


def relative_geometry_constraints(phase: Phase) -> tuple[RelativeGeometryConstraint, ...]:
    """Return the phase's relative-geometry constraints."""
    return tuple(
        constraint
        for constraint in phase.constraints
        if is_relative_geometry_constraint(constraint)
    )


def apply_relative_geometry_constraint(
    asset_phase: Any,
    constraint: RelativeGeometryConstraint,
    position_indices: tuple[int, ...] = (0, 1, 2),
    *,
    time_index: int = 6,
    solar_direction_table: Any | None = None,
) -> None:
    """Compatibility wrapper for relative-geometry ``apply`` methods."""

    class _PositionLayout:
        name = "relative_cartesian"
        time_column = int(time_index)

        @staticmethod
        def state_indices(group: str) -> tuple[int, ...]:
            if group != "position":
                raise KeyError(group)
            return tuple(int(index) for index in position_indices)

    constraint.apply(
        asset_phase,
        ConstraintContext(
            vector_functions=vf,
            layout=_PositionLayout(),
            is_relative_phase=True,
            solar_direction_table=solar_direction_table,
        ),
    )


def relative_constraint_report_rows(
    *,
    phase_name: str,
    constraint: RelativeGeometryConstraint,
    phase_traj: np.ndarray,
    solar_direction_at: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[dict[str, float | str | bool]]:
    """Compatibility wrapper for relative-geometry ``report`` methods."""
    trajectory = np.asarray(phase_traj, dtype=float)
    return constraint.report(
        ConstraintReport(
            phase_name=phase_name,
            phase_trajectory=trajectory,
            native_trajectory=trajectory,
            relative_trajectory=trajectory,
            layout=None,
            mu_m3ps2=0.0,
            solar_direction_at=solar_direction_at,
        )
    )
