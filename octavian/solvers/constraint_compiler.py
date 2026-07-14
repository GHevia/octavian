"""Constraint helpers for the composable ASSET backend.

This module owns the translation-oriented pieces for user-facing constraint
objects. It keeps constraint lookup, orbital-element ASSET expressions, terminal
post-burn shell handling, and solved-trajectory constraint reports out of the
main composable solver flow.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .._asset import vf
from ..astro.kepler import cartesian_to_classic
from ..astro.types import as_vec3
from ..constraints import Constraint, OrbitalElementConstraint, Position, State
from ..links import impulsive as impulsive_link
from ..phase import Phase
from ..variables import ImpulsiveDeltaV


def get_constraint_of_type(
    phase: Phase, constraint_type: type[Constraint], where: str
) -> Constraint | None:
    """Return the first constraint of ``constraint_type`` at a phase boundary."""
    loc = "Front" if where.lower().startswith("f") else "Back"
    for constraint in getattr(phase, "constraints", []) or []:
        if isinstance(constraint, constraint_type) and getattr(constraint, "where", "") == loc:
            return constraint
    return None


def get_state_constraint(phase: Phase, where: str) -> Constraint | None:
    """Return the first state constraint at a phase boundary, if present."""
    return get_constraint_of_type(phase, State, where)


def get_position_constraint(phase: Phase, where: str) -> Constraint | None:
    """Return the first position constraint at a phase boundary, if present."""
    return get_constraint_of_type(phase, Position, where)


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
        if not isinstance(constraint, OrbitalElementConstraint):
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


@dataclass(frozen=True)
class OrbitalElementExpressions:
    """ASSET scalar expressions used by orbital-element constraints."""

    semi_major_axis_m: Any
    eccentricity_sq: Any
    inclination_cosine: Any


def orbital_element_expressions(mu_m3ps2: float) -> OrbitalElementExpressions:
    """Build ASSET expressions for the orbital elements used by Octavian."""
    arguments = vf.Arguments(6)
    position_vec, velocity_vec = arguments.tolist([(0, 3), (3, 3)])
    angular_momentum_vec = position_vec.cross(velocity_vec)
    radius_norm = position_vec.norm()
    speed_norm = velocity_vec.norm()
    specific_energy = 0.5 * (speed_norm**2) - float(mu_m3ps2) / radius_norm
    angular_momentum_sq = angular_momentum_vec.dot(angular_momentum_vec)
    return OrbitalElementExpressions(
        semi_major_axis_m=-0.5 * float(mu_m3ps2) / specific_energy,
        eccentricity_sq=1.0
        + (2.0 * specific_energy * angular_momentum_sq) / (float(mu_m3ps2) ** 2),
        inclination_cosine=angular_momentum_vec.normalized()[2],
    )


def apply_orbital_element_constraint(
    asset_phase: Any, constraint: OrbitalElementConstraint, mu_m3ps2: float
) -> None:
    """Apply one orbital-element constraint to an ASSET phase.

    Equality constraints are used when the user did not specify a tolerance.
    With a tolerance, the compiler emits paired upper/lower inequality
    constraints around the target element value.
    """
    expressions = orbital_element_expressions(mu_m3ps2)
    where = getattr(constraint, "where", "Path")
    values = constraint.value

    if constraint.kind == "semi_major_axis":
        target = float(values["a_m"])
        tolerance = values["tol_m"]
        if tolerance is None:
            asset_phase.addEqualCon(
                where, vf.stack([expressions.semi_major_axis_m - target]), range(0, 6)
            )
            return
        tolerance = float(tolerance)
        asset_phase.addInequalCon(
            where,
            vf.stack([expressions.semi_major_axis_m - (target + tolerance)]),
            range(0, 6),
        )
        asset_phase.addInequalCon(
            where,
            vf.stack([(target - tolerance) - expressions.semi_major_axis_m]),
            range(0, 6),
        )
        return

    if constraint.kind == "eccentricity":
        target = float(values["e"])
        tolerance = values["tol"]
        if tolerance is None:
            asset_phase.addEqualCon(
                where, vf.stack([expressions.eccentricity_sq - target**2]), range(0, 6)
            )
            return
        tolerance = float(tolerance)
        asset_phase.addInequalCon(
            where,
            vf.stack([expressions.eccentricity_sq - (target + tolerance) ** 2]),
            range(0, 6),
        )
        asset_phase.addInequalCon(
            where,
            vf.stack([(target - tolerance) ** 2 - expressions.eccentricity_sq]),
            range(0, 6),
        )
        return

    if constraint.kind == "inclination_deg":
        target_deg = float(values["inc_deg"])
        tolerance_deg = values["tol_deg"]
        target_cosine = float(np.cos(np.deg2rad(target_deg)))
        if tolerance_deg is None:
            asset_phase.addEqualCon(
                where, vf.stack([expressions.inclination_cosine - target_cosine]), range(0, 6)
            )
            return
        tolerance_deg = float(tolerance_deg)
        upper_cosine = float(np.cos(np.deg2rad(target_deg - tolerance_deg)))
        lower_cosine = float(np.cos(np.deg2rad(target_deg + tolerance_deg)))
        asset_phase.addInequalCon(
            where,
            vf.stack([expressions.inclination_cosine - upper_cosine]),
            range(0, 6),
        )
        asset_phase.addInequalCon(
            where,
            vf.stack([lower_cosine - expressions.inclination_cosine]),
            range(0, 6),
        )
        return

    raise ValueError(f"Unsupported orbital-element constraint kind: {constraint.kind!r}")


def boundary_state_from_traj(traj: np.ndarray, where: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract a Cartesian boundary state from one phase trajectory."""
    row = np.asarray(traj[0], dtype=float) if where == "Front" else np.asarray(traj[-1], dtype=float)
    return as_vec3(row[0:3]), as_vec3(row[3:6])


def orbital_constraint_report_row(
    *,
    phase_name: str,
    constraint: OrbitalElementConstraint,
    phase_traj: np.ndarray,
    mu_m3ps2: float,
) -> dict[str, float | str | bool]:
    """Summarize one orbital-element constraint against the solved trajectory."""
    position_m, velocity_mps = boundary_state_from_traj(phase_traj, constraint.where)
    actual_elements = cartesian_to_classic(
        r_m=position_m,
        v_mps=velocity_mps,
        mu_m3ps2=mu_m3ps2,
    )

    if constraint.kind == "semi_major_axis":
        target = float(constraint.a_m)
        tolerance = constraint.tol_m
        actual = float(actual_elements["a_m"])
    elif constraint.kind == "eccentricity":
        target = float(constraint.e)
        tolerance = constraint.tol
        actual = float(actual_elements["e"])
    elif constraint.kind == "inclination_deg":
        target = float(constraint.inc_deg)
        tolerance = constraint.tol_deg
        actual = float(actual_elements["inc_deg"])
    else:
        raise ValueError(f"Unsupported orbital-element constraint kind: {constraint.kind!r}")

    error = actual - target
    base_tolerance = float(tolerance) if tolerance is not None else 1e-6
    report_tolerance = base_tolerance + max(1e-9, 1e-7 * max(1.0, abs(target)))
    satisfied = abs(error) <= report_tolerance
    return {
        "phase": phase_name,
        "where": constraint.where,
        "family": constraint.family,
        "constraint": constraint.kind,
        "target": target,
        "tolerance": base_tolerance,
        "actual": actual,
        "error": error,
        "satisfied": satisfied,
    }


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
