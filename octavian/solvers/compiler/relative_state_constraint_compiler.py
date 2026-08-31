"""Compatibility helpers for native relative-state constraints."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._asset import vf
from ...constraints import (
    Constraint,
    RelativeOrbitalElementConstraint,
    RelativeOrbitalElementsConstraint,
    RelativeStateComponent,
)
from ...coordinates import StateLayout
from ...phase import Phase
from ..constraint_compiler import ConstraintContext, ConstraintReport

NativeRelativeConstraint = (
    RelativeStateComponent | RelativeOrbitalElementConstraint | RelativeOrbitalElementsConstraint
)


def is_native_relative_constraint(constraint: Constraint) -> bool:
    """Return whether ``constraint`` targets a native relative state."""
    return constraint.family in {"relative_state", "relative_orbital_element"}


def native_relative_constraints(
    phase: Phase,
) -> tuple[NativeRelativeConstraint, ...]:
    """Return direct RIC and relative-element constraints on a phase."""
    return tuple(
        constraint
        for constraint in phase.constraints
        if is_native_relative_constraint(constraint)
    )


def apply_native_relative_constraint(
    asset_phase: Any,
    constraint: NativeRelativeConstraint,
    layout: StateLayout,
) -> None:
    """Compatibility wrapper for native relative constraint ``apply`` methods."""
    constraint.apply(
        asset_phase,
        ConstraintContext(
            vector_functions=vf,
            layout=layout,
            is_relative_phase=True,
        ),
    )


def relative_state_constraint_report_rows(
    *,
    phase_name: str,
    constraint: NativeRelativeConstraint,
    native_trajectory: np.ndarray,
    layout: StateLayout,
) -> list[dict[str, float | str | bool]]:
    """Compatibility wrapper for native relative constraint ``report`` methods."""
    trajectory = np.asarray(native_trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] <= layout.time_column:
        raise ValueError("native_trajectory does not match the supplied layout")
    return constraint.report(
        ConstraintReport(
            phase_name=phase_name,
            phase_trajectory=trajectory,
            native_trajectory=trajectory,
            relative_trajectory=None,
            layout=layout,
            mu_m3ps2=0.0,
        )
    )
