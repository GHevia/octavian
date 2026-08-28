"""Constraint helpers for the composable ASSET backend.

This module supplies transient application/report contexts, compatibility
wrappers, lookup helpers, and terminal post-burn shell handling. Concrete
constraint formulas live with their declarations in ``octavian.constraints``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._asset import vf
from ..cislunar import CR3BPSystem
from ..constraints import (
    Constraint,
    JacobiConstant,
    OrbitalElementConstraint,
    PeriodicState,
    StateComponent,
)
from ..coordinates import StateLayout
from ..links import impulsive as impulsive_link
from ..phase import Phase
from ..variables import ImpulsiveDeltaV


@dataclass(frozen=True, slots=True)
class ConstraintContext:
    """Phase-specific inputs shared by concrete ``Constraint.apply`` methods."""

    vector_functions: Any = vf
    layout: Any = None
    declared_phase: Any = None
    phase_index: int = 0
    mu_m3ps2: float = 0.0
    cr3bp_system: Any | None = None
    is_relative_phase: bool = False
    relative_expressions: Any | None = None
    third_body_tables: dict[str, Any] | None = None
    solar_direction_table: Any | None = None
    solar_position_table: Any | None = None

    def __post_init__(self) -> None:
        if self.third_body_tables is None:
            object.__setattr__(self, "third_body_tables", {})


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    """Solved phase data supplied to polymorphic constraint reports."""

    phase_name: str
    phase_trajectory: np.ndarray
    native_trajectory: np.ndarray
    relative_trajectory: np.ndarray | None
    layout: Any
    mu_m3ps2: float
    cr3bp_system: Any | None = None
    solar_direction_at: Callable[[np.ndarray], np.ndarray] | None = None


def apply_constraint(
    asset_phase: Any,
    constraint: Constraint,
    context: ConstraintContext,
) -> None:
    """Apply one declaration through its concrete single-dispatch method."""
    constraint.apply(asset_phase, context)


def apply_constraints(
    asset_phase: Any,
    constraints: Iterable[Constraint],
    context: ConstraintContext,
) -> None:
    """Apply phase constraints without a central concrete-type registry."""
    for constraint in constraints:
        constraint.apply(asset_phase, context)


def get_state_constraint(phase: Phase, where: str) -> Constraint | None:
    """Return the first state constraint at a phase boundary, if present."""
    location = "Front" if where.lower().startswith("f") else "Back"
    return next(
        (
            constraint
            for constraint in getattr(phase, "constraints", []) or []
            if getattr(constraint, "kind", "") == "state"
            and getattr(constraint, "where", "") == location
        ),
        None,
    )


def get_position_constraint(phase: Phase, where: str) -> Constraint | None:
    """Return the first position constraint at a phase boundary, if present."""
    location = "Front" if where.lower().startswith("f") else "Back"
    return next(
        (
            constraint
            for constraint in getattr(phase, "constraints", []) or []
            if getattr(constraint, "kind", "") == "position"
            and getattr(constraint, "where", "") == location
        ),
        None,
    )


def orbital_element_constraints(
    phase: Phase, where: str | None = None
) -> tuple[OrbitalElementConstraint, ...]:
    """Return orbital-element constraints, optionally filtered by boundary.

    ``where`` accepts the same user-facing spelling variants used elsewhere in
    the composable backend: front/start, back/end, and path/all.
    """
    if where is None:
        expected_where = None
    else:
        normalized = where.lower()
        expected_where = (
            "Front"
            if normalized.startswith("f")
            else "Back" if normalized.startswith("b") else "Path"
        )

    matches: list[OrbitalElementConstraint] = []
    for constraint in getattr(phase, "constraints", []) or []:
        if getattr(constraint, "family", "") != "orbital_element":
            continue
        if expected_where is not None and constraint.where != expected_where:
            continue
        matches.append(constraint)
    return tuple(matches)


def state_boundary_value(constraint: Constraint | None) -> Any:
    """Return the boundary-state payload stored by a state constraint."""
    if constraint is None:
        return None
    return getattr(constraint, "value", {}).get("x")


def state_groups(constraint: Constraint | None) -> tuple[str, ...]:
    """Return state groups constrained by a state constraint."""
    if constraint is None:
        return tuple()
    groups = getattr(constraint, "value", {}).get("groups", ("R", "V"))
    return tuple(str(group) for group in groups)


def position_boundary_value(constraint: Constraint | None) -> np.ndarray | None:
    """Return the Cartesian position payload stored by a position constraint."""
    if constraint is None:
        return None
    return np.asarray(constraint.value, dtype=float).reshape(3)


def apply_state_component_constraint(
    asset_phase: Any,
    constraint: StateComponent,
    layout: StateLayout,
) -> None:
    """Compatibility wrapper for ``StateComponent.apply``."""
    constraint.apply(
        asset_phase,
        ConstraintContext(vector_functions=vf, layout=layout),
    )


def apply_periodic_state_constraint(
    asset_phase: Any,
    constraint: PeriodicState,
    layout: StateLayout,
) -> None:
    """Compatibility wrapper for ``PeriodicState.apply``."""
    constraint.apply(
        asset_phase,
        ConstraintContext(vector_functions=vf, layout=layout),
    )


def apply_jacobi_constant_constraint(
    asset_phase: Any,
    constraint: JacobiConstant,
    system: CR3BPSystem,
) -> None:
    """Compatibility wrapper for ``JacobiConstant.apply``."""
    constraint.apply(
        asset_phase,
        ConstraintContext(vector_functions=vf, cr3bp_system=system),
    )


def make_terminal_shell(last_phase: Phase) -> tuple[Phase, Phase] | None:
    """Create an internal post-burn shell phase for terminal orbital targets.

    A back-boundary impulse changes terminal velocity after the phase dynamics
    finish. Orbital-element constraints should see that post-impulse state, so
    the compiler inserts a short shell phase with a front impulse and moves the
    terminal orbital-element constraints onto that shell.
    """
    if not _has_terminal_post_burn_orbital_target(last_phase):
        return None

    moved_constraints = list(orbital_element_constraints(last_phase, "Back"))
    compile_last = replace(
        last_phase,
        constraints=[
            constraint
            for constraint in (last_phase.constraints or [])
            if constraint not in moved_constraints
        ],
        variables=[
            variable
            for variable in (last_phase.variables or [])
            if not (
                isinstance(variable, ImpulsiveDeltaV)
                and getattr(variable, "where", "") == "Back"
            )
        ],
    )
    shell_phase = Phase(
        name=f"{last_phase.name}_post_burn",
        mode="coast",
        spacecraft=last_phase.spacecraft,
        dynamics=last_phase.dynamics,
        previous=last_phase,
        link=impulsive_link(name="post_burn"),
        tof_bounds_s=(0.1, 1.0),
        tof_is_relative=True,
        constraints=[replace(constraint, where="Front") for constraint in moved_constraints],
        variables=[ImpulsiveDeltaV(where="Front")],
    )
    return compile_last, shell_phase


def apply_orbital_element_constraint(
    asset_phase: Any, constraint: OrbitalElementConstraint, mu_m3ps2: float
) -> None:
    """Compatibility wrapper for concrete orbital-element ``apply`` methods."""
    constraint.apply(
        asset_phase,
        ConstraintContext(vector_functions=vf, mu_m3ps2=float(mu_m3ps2)),
    )


def orbital_constraint_report_row(
    *,
    phase_name: str,
    constraint: OrbitalElementConstraint,
    phase_traj: np.ndarray,
    mu_m3ps2: float,
) -> dict[str, float | str | bool]:
    """Compatibility wrapper for orbital-element ``report`` methods."""
    trajectory = np.asarray(phase_traj, dtype=float)
    rows = constraint.report(
        ConstraintReport(
            phase_name=phase_name,
            phase_trajectory=trajectory,
            native_trajectory=trajectory,
            relative_trajectory=trajectory,
            layout=None,
            mu_m3ps2=float(mu_m3ps2),
        )
    )
    if not rows:
        raise ValueError("Orbital-element reporting requires a Front or Back constraint")
    return rows[0]


def jacobi_constraint_report_row(
    *,
    phase_name: str,
    constraint: JacobiConstant,
    phase_traj: np.ndarray,
    system: CR3BPSystem,
) -> dict[str, float | str | bool]:
    """Compatibility wrapper for ``JacobiConstant.report``."""
    trajectory = np.asarray(phase_traj, dtype=float)
    rows = constraint.report(
        ConstraintReport(
            phase_name=phase_name,
            phase_trajectory=trajectory,
            native_trajectory=trajectory,
            relative_trajectory=trajectory,
            layout=None,
            mu_m3ps2=0.0,
            cr3bp_system=system,
        )
    )
    if not rows:
        raise ValueError("Jacobi reporting requires a CR3BP system")
    return rows[0]


def _has_terminal_post_burn_orbital_target(phase: Phase) -> bool:
    """Return whether a terminal shell is needed after a back impulse."""
    has_back_impulse = any(
        isinstance(variable, ImpulsiveDeltaV) and getattr(variable, "where", "") == "Back"
        for variable in getattr(phase, "variables", []) or []
    )
    has_back_event = any(
        getattr(event, "kind", "") == "impulse" and getattr(event, "where", "") == "Back"
        for event in getattr(phase, "events", []) or []
    )
    return (has_back_impulse or has_back_event) and bool(
        orbital_element_constraints(phase, "Back")
    )
