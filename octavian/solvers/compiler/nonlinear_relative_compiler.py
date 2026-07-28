"""Compiler boundary for exact nonlinear relative-motion phases.

The nonlinear ODE propagates two ordinary absolute Cartesian states.  This
module is the single place that presents that 12-state implementation as the
public six-state RIC model used by constraints, objectives, and results.
Keeping the coordinate algebra here prevents force models from being mixed
with a CWH linearization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._asset import vf
from ...astro.types import as_vec3
from ...constraints import (
    ApproachCone,
    KeepOutSphere,
    LightingAngle,
    SolarPhaseAngle,
)
from ...dynamics import (
    ThirdBodyTable,
    _gravity_acceleration,
    gravity_acceleration_components,
)
from ...phase import Phase
from ...relative import NonlinearRelative
from ...relative.transforms import inertial_to_relative_state, ric_basis
from ...specs import BoundaryState
from ..third_bodies import phase_perturbations, tables_for_phase
from .relative_constraint_compiler import RelativeGeometryConstraint

COUPLED_ARGUMENT_INDICES = tuple(range(13))


@dataclass(frozen=True, slots=True)
class RelativeStateExpressions:
    """Symbolic state views derived from a coupled absolute state."""

    position: Any
    velocity: Any
    chief_position: Any
    chief_velocity: Any
    deputy_position: Any
    deputy_velocity: Any
    radial_axis: Any
    in_track_axis: Any
    cross_track_axis: Any
    time: Any


def model_for_phase(phase: Phase) -> NonlinearRelative | None:
    """Return the phase's exact relative model, when configured."""
    dynamics = phase.dynamics
    model = dynamics.model if dynamics is not None else None
    return model if isinstance(model, NonlinearRelative) else None


def relative_state_expressions(
    phase: Phase,
    third_body_tables: dict[str, ThirdBodyTable],
) -> RelativeStateExpressions:
    """Build the instantaneous RIC state from 12 absolute states and time."""
    dynamics = phase.dynamics
    if dynamics is None or model_for_phase(phase) is None:
        raise TypeError("Exact RIC expressions require Dynamics.relative(...)")
    arguments = vf.Arguments(13)
    chief_position = arguments.head(3)
    chief_velocity = arguments.segment(3, 3)
    deputy_position = arguments.segment(6, 3)
    deputy_velocity = arguments.segment(9, 3)
    time = arguments[12]

    radial = chief_position.normalized()
    angular_momentum = chief_position.cross(chief_velocity)
    cross_track = angular_momentum.normalized()
    in_track = cross_track.cross(radial)

    def rotate_to_ric(vector):
        return vf.stack(
            [
                radial.dot(vector),
                in_track.dot(vector),
                cross_track.dot(vector),
            ]
        )

    relative_position = rotate_to_ric(deputy_position - chief_position)
    inertial_velocity_difference_ric = rotate_to_ric(
        deputy_velocity - chief_velocity
    )
    perturbations = phase_perturbations(phase)
    chief_acceleration = _gravity_acceleration(
        chief_position,
        mu_m3ps2=float(dynamics.mu_m3ps2),
        include_j2=bool(perturbations.j2),
        central_body_radius_m=float(dynamics.central_body_radius_m),
        j2_coefficient=float(dynamics.j2_coefficient),
        time_var=time,
        third_body_tables=tables_for_phase(phase, third_body_tables),
    )
    frame_rate_ric = vf.stack(
        [
            chief_position.norm()
            * cross_track.dot(chief_acceleration)
            / angular_momentum.norm(),
            chief_position[0] * 0.0,
            angular_momentum.norm() / chief_position.dot(chief_position),
        ]
    )
    relative_velocity = inertial_velocity_difference_ric - frame_rate_ric.cross(
        relative_position
    )
    return RelativeStateExpressions(
        position=relative_position,
        velocity=relative_velocity,
        chief_position=chief_position,
        chief_velocity=chief_velocity,
        deputy_position=deputy_position,
        deputy_velocity=deputy_velocity,
        radial_axis=radial,
        in_track_axis=in_track,
        cross_track_axis=cross_track,
        time=time,
    )


def fix_initial_chief(asset_phase: Any, model: NonlinearRelative) -> None:
    """Fix the absolute chief state that defines the public RIC frame."""
    chief = model.chief_initial_state_eci
    asset_phase.addBoundaryValue(
        "Front",
        ["ChiefR", "ChiefV"],
        np.hstack([chief.r_m, chief.v_mps]),
    )


def apply_position_boundary(
    asset_phase: Any,
    where: str,
    target_position_m: np.ndarray,
    expressions: RelativeStateExpressions,
) -> None:
    """Apply a public RIC position constraint to a coupled phase."""
    residual = expressions.position - as_vec3(target_position_m)
    asset_phase.addEqualCon(where, residual, COUPLED_ARGUMENT_INDICES)


def apply_state_boundary(
    asset_phase: Any,
    where: str,
    state: BoundaryState,
    groups: Sequence[str],
    expressions: RelativeStateExpressions,
) -> None:
    """Apply selected public RIC state groups to a coupled phase."""
    for group in groups:
        if group == "R":
            residual = expressions.position - as_vec3(state.r_m)
        elif group == "V":
            residual = expressions.velocity - as_vec3(state.v_mps)
        elif group in {"t", "time"}:
            continue
        else:
            raise ValueError(f"Unsupported State.groups element: {group!r}")
        asset_phase.addEqualCon(where, residual, COUPLED_ARGUMENT_INDICES)


def apply_minimum_range(
    asset_phase: Any,
    where: str,
    minimum_range_m: float,
    expressions: RelativeStateExpressions,
) -> None:
    """Keep deputy range above a lower bound in the public RIC frame."""
    violation = float(minimum_range_m) ** 2 - expressions.position.dot(
        expressions.position
    )
    asset_phase.addInequalCon(
        where,
        vf.stack([violation]),
        COUPLED_ARGUMENT_INDICES,
    )


def apply_geometry_constraint(
    asset_phase: Any,
    constraint: RelativeGeometryConstraint,
    expressions: RelativeStateExpressions,
    *,
    solar_position_table: Any | None = None,
) -> None:
    """Compile relative geometry against exact coupled-state expressions."""
    position = expressions.position
    where = constraint.where

    if isinstance(constraint, KeepOutSphere):
        offset = position - np.asarray(constraint.center_m, dtype=float)
        violation = constraint.radius_m**2 - offset.dot(offset)
        asset_phase.addInequalCon(
            where, vf.stack([violation]), COUPLED_ARGUMENT_INDICES
        )
        return

    if isinstance(constraint, ApproachCone):
        offset = position - np.asarray(constraint.vertex_m, dtype=float)
        axial_distance = offset.dot(np.asarray(constraint.axis, dtype=float))
        cosine_sq = float(np.cos(np.deg2rad(constraint.half_angle_deg)) ** 2)
        cone_violation = cosine_sq * offset.dot(offset) - axial_distance**2
        asset_phase.addInequalCon(
            where, vf.stack([cone_violation]), COUPLED_ARGUMENT_INDICES
        )
        asset_phase.addInequalCon(
            where, vf.stack([-axial_distance]), COUPLED_ARGUMENT_INDICES
        )
        return

    if isinstance(constraint, LightingAngle):
        offset = position - np.asarray(constraint.origin_m, dtype=float)
        direction = np.asarray(constraint.sun_direction, dtype=float)
        _apply_angle_bounds(
            asset_phase,
            where,
            offset,
            direction,
            constraint.min_angle_deg,
            constraint.max_angle_deg,
        )
        return

    if isinstance(constraint, SolarPhaseAngle):
        if solar_position_table is None:
            raise ValueError(
                "SolarPhaseAngle requires a SPICE-derived ECI Sun position table"
            )
        offset_ric = position - np.asarray(constraint.origin_m, dtype=float)
        sun_line_eci = solar_position_table(expressions.time) - expressions.chief_position
        sun_line_ric = vf.stack(
            [
                expressions.radial_axis.dot(sun_line_eci),
                expressions.in_track_axis.dot(sun_line_eci),
                expressions.cross_track_axis.dot(sun_line_eci),
            ]
        )
        _apply_angle_bounds(
            asset_phase,
            where,
            offset_ric,
            sun_line_ric,
            constraint.min_angle_deg,
            constraint.max_angle_deg,
        )
        return

    raise TypeError(f"Unsupported relative geometry constraint {type(constraint).__name__}")


def add_velocity_objective(
    asset_phase: Any,
    where: str,
    target_velocity_mps: np.ndarray,
    weight: float,
    velocity_unit_mps: float,
    expressions: RelativeStateExpressions,
) -> None:
    """Minimize an impulsive change from a target public RIC velocity."""
    difference = expressions.velocity - as_vec3(target_velocity_mps)
    magnitude = vf.sqrt(difference.dot(difference))
    asset_phase.addStateObjective(
        where,
        float(weight) * magnitude,
        list(COUPLED_ARGUMENT_INDICES),
        [],
        [],
        AutoScale=1.0 / float(velocity_unit_mps),
    )


def coupled_trajectory_rvt(
    raw_trajectory: np.ndarray,
    phase: Phase,
    third_body_tables: dict[str, ThirdBodyTable],
) -> np.ndarray:
    """Convert ``[chief ECI, deputy ECI, t]`` rows to public RIC rows."""
    raw = np.asarray(raw_trajectory, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 13:
        raise ValueError("Coupled relative trajectory must contain 13 columns")
    dynamics = phase.dynamics
    if dynamics is None or model_for_phase(phase) is None:
        raise TypeError("Coupled extraction requires Dynamics.relative(...)")
    perturbations = phase_perturbations(phase)
    body_tables = tables_for_phase(phase, third_body_tables)
    converted = np.empty((raw.shape[0], 7), dtype=float)
    for index, row in enumerate(raw):
        chief = BoundaryState(row[0:3], row[3:6])
        deputy = BoundaryState(row[6:9], row[9:12])
        chief_acceleration = gravity_acceleration_components(
            chief.r_m,
            time_s=float(row[12]),
            mu_m3ps2=float(dynamics.mu_m3ps2),
            include_j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=body_tables,
        )
        relative = inertial_to_relative_state(
            chief,
            deputy,
            chief_acceleration_mps2=chief_acceleration,
        )
        converted[index] = np.hstack(
            [relative.r_m, relative.v_mps, float(row[12])]
        )
    return converted


def solar_directions_ric(
    raw_trajectory: np.ndarray,
    solar_table: Any,
) -> np.ndarray:
    """Return chief-to-Sun unit directions in instantaneous RIC."""
    raw = np.asarray(raw_trajectory, dtype=float)
    sun_positions = solar_table.sun_position_at(raw[:, 12])
    directions = np.empty((raw.shape[0], 3), dtype=float)
    for index, (row, sun_position) in enumerate(
        zip(raw, sun_positions, strict=True)
    ):
        line_eci = sun_position - row[0:3]
        line_ric = ric_basis(row[0:3], row[3:6]) @ line_eci
        directions[index] = line_ric / np.linalg.norm(line_ric)
    return directions


def _apply_angle_bounds(
    asset_phase: Any,
    where: str,
    vector: Any,
    direction: Any,
    minimum_angle_deg: float,
    maximum_angle_deg: float,
) -> None:
    vector_norm = vector.norm()
    direction_norm = direction.norm() if hasattr(direction, "norm") else 1.0
    projection = vector.dot(direction)
    scale = vector_norm * direction_norm
    minimum_cosine = float(np.cos(np.deg2rad(minimum_angle_deg)))
    maximum_cosine = float(np.cos(np.deg2rad(maximum_angle_deg)))
    asset_phase.addInequalCon(
        where,
        vf.stack([maximum_cosine * scale - projection]),
        COUPLED_ARGUMENT_INDICES,
    )
    asset_phase.addInequalCon(
        where,
        vf.stack([projection - minimum_cosine * scale]),
        COUPLED_ARGUMENT_INDICES,
    )
