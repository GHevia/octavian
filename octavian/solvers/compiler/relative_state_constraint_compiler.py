"""Compile and report constraints on native relative-state representations."""

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

NativeRelativeConstraint = (
    RelativeStateComponent | RelativeOrbitalElementConstraint | RelativeOrbitalElementsConstraint
)

_RIC_COMPONENT_GROUPS = {
    "R": ("position", 0),
    "I": ("position", 1),
    "C": ("position", 2),
    "Rdot": ("velocity", 0),
    "Idot": ("velocity", 1),
    "Cdot": ("velocity", 2),
}


def is_native_relative_constraint(constraint: Constraint) -> bool:
    """Return whether ``constraint`` targets a native relative state."""
    return isinstance(
        constraint,
        (
            RelativeStateComponent,
            RelativeOrbitalElementConstraint,
            RelativeOrbitalElementsConstraint,
        ),
    )


def native_relative_constraints(
    phase: Phase,
) -> tuple[NativeRelativeConstraint, ...]:
    """Return direct RIC and relative-element constraints on a phase."""
    return tuple(
        constraint for constraint in phase.constraints if is_native_relative_constraint(constraint)
    )


def apply_native_relative_constraint(
    asset_phase: Any,
    constraint: NativeRelativeConstraint,
    layout: StateLayout,
) -> None:
    """Apply one constraint directly to its native ASSET state variable.

    Raises:
        ValueError: If the selected propagation layout does not natively store
            the requested RIC or relative-element representation.
    """
    if isinstance(constraint, RelativeStateComponent):
        group, offset = _RIC_COMPONENT_GROUPS[constraint.component]
        try:
            state_index = layout.state_indices(group)[offset]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "RIC component constraints require CWH, 'nonlinear_ric', or "
                "'coupled_ric' propagation; the selected layout is "
                f"{layout.name!r}"
            ) from exc
        _apply_scalar_target(
            asset_phase,
            where=constraint.where,
            state_index=state_index,
            target=constraint.target,
            tolerance=constraint.tolerance,
        )
        return

    representation = str(constraint.representation)
    expected_layout = (
        "damico_relative_elements" if representation == "damico" else "classical_relative_elements"
    )
    if layout.name != expected_layout:
        raise ValueError(
            f"{representation!r} constraints require a matching native "
            f"relative-element propagation mode; selected layout is {layout.name!r}"
        )
    if isinstance(constraint, RelativeOrbitalElementsConstraint):
        indices = layout.state_indices("relative_elements")
        asset_phase.addBoundaryValue(
            constraint.where,
            list(indices),
            np.asarray(constraint.elements, dtype=float),
        )
        return

    state_index = layout.state_indices(constraint.element)[0]
    _apply_scalar_target(
        asset_phase,
        where=constraint.where,
        state_index=state_index,
        target=constraint.target,
        tolerance=constraint.tolerance,
    )


def relative_state_constraint_report_rows(
    *,
    phase_name: str,
    constraint: NativeRelativeConstraint,
    native_trajectory: np.ndarray,
    layout: StateLayout,
) -> list[dict[str, float | str | bool]]:
    """Evaluate a native relative-state constraint against solver output."""
    trajectory = np.asarray(native_trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] <= layout.time_column:
        raise ValueError("native_trajectory does not match the supplied layout")

    if isinstance(constraint, RelativeStateComponent):
        group, offset = _RIC_COMPONENT_GROUPS[constraint.component]
        index = layout.state_indices(group)[offset]
        return [
            _scalar_report_row(
                phase_name=phase_name,
                constraint=constraint,
                name=f"ric_{constraint.component}",
                target=constraint.target,
                tolerance=constraint.tolerance,
                values=_values_at_location(
                    trajectory[:, index],
                    constraint.where,
                ),
            )
        ]

    if isinstance(constraint, RelativeOrbitalElementConstraint):
        index = layout.state_indices(constraint.element)[0]
        return [
            _scalar_report_row(
                phase_name=phase_name,
                constraint=constraint,
                name=f"{constraint.representation}_{constraint.element}",
                target=constraint.target,
                tolerance=constraint.tolerance,
                values=_values_at_location(
                    trajectory[:, index],
                    constraint.where,
                ),
            )
        ]

    indices = layout.state_indices("relative_elements")
    names = layout.state_names
    rows: list[dict[str, float | str | bool]] = []
    for index, target in zip(indices, constraint.elements, strict=True):
        rows.append(
            _scalar_report_row(
                phase_name=phase_name,
                constraint=constraint,
                name=f"{constraint.representation}_{names[index]}",
                target=float(target),
                tolerance=None,
                values=_values_at_location(
                    trajectory[:, index],
                    constraint.where,
                ),
            )
        )
    return rows


def _apply_scalar_target(
    asset_phase: Any,
    *,
    where: str,
    state_index: int,
    target: float,
    tolerance: float | None,
) -> None:
    """Compile an equality or symmetric tolerance band on one state."""
    variable = vf.Arguments(1).tolist()[0]
    residual = variable - float(target)
    if tolerance is None:
        asset_phase.addEqualCon(where, residual, [int(state_index)])
        return
    asset_phase.addInequalCon(
        where,
        vf.stack([residual - float(tolerance)]),
        [int(state_index)],
    )
    asset_phase.addInequalCon(
        where,
        vf.stack([-residual - float(tolerance)]),
        [int(state_index)],
    )


def _values_at_location(values: np.ndarray, where: str) -> np.ndarray:
    """Select front, back, or all path values."""
    if where == "Front":
        return values[0:1]
    if where == "Back":
        return values[-1:]
    return values


def _scalar_report_row(
    *,
    phase_name: str,
    constraint: NativeRelativeConstraint,
    name: str,
    target: float,
    tolerance: float | None,
    values: np.ndarray,
) -> dict[str, float | str | bool]:
    """Build a standard constraint-report row for one scalar target."""
    declared_tolerance = 0.0 if tolerance is None else float(tolerance)
    errors = np.asarray(values, dtype=float) - float(target)
    worst_index = int(np.argmax(np.abs(errors)))
    actual = float(np.asarray(values, dtype=float)[worst_index])
    error = float(errors[worst_index])
    numerical_tolerance = max(
        declared_tolerance,
        1.0e-9 * max(1.0, abs(float(target))),
    )
    return {
        "phase": phase_name,
        "where": constraint.where,
        "family": constraint.family,
        "constraint": name,
        "target": float(target),
        "tolerance": declared_tolerance,
        "actual": actual,
        "error": error,
        "satisfied": bool(abs(error) <= numerical_tolerance),
    }
