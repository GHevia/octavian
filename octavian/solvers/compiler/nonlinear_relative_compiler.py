"""Compiler boundary for exact nonlinear relative-motion phases.

The nonlinear ODE propagates two ordinary absolute Cartesian states.  This
module is the single place that presents that 12-state implementation as the
public six-state RIC model used by constraints, objectives, and results.
Keeping the coordinate algebra here prevents force models from being mixed
with a CWH linearization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._asset import vf
from ...astro.types import as_vec3
from ...coordinates import StateLayout
from ...dynamics import (
    ThirdBodyTable,
    _translational_acceleration,
    translational_acceleration_components,
)
from ...forces import EARTH_EXPONENTIAL_ATMOSPHERE
from ...phase import Phase
from ...relative import (
    NonlinearRelative,
    RelativePropagationMode,
    classical_relative_orbital_elements_to_relative_state,
    propagate_two_body_state,
    relative_orbital_elements_to_relative_state,
)
from ...relative.transforms import (
    inertial_to_relative_state,
    relative_to_inertial_state,
    ric_basis,
)
from ...specs import BoundaryState
from ..constraint_compiler import ConstraintContext
from ..third_bodies import phase_perturbations, sun_table_for_phase, tables_for_phase
from .relative_constraint_compiler import RelativeGeometryConstraint


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
    argument_indices: tuple[int, ...]


def model_for_phase(phase: Phase) -> NonlinearRelative | None:
    """Return the phase's exact relative model, when configured."""
    dynamics = phase.dynamics
    model = dynamics.model if dynamics is not None else None
    return model if isinstance(model, NonlinearRelative) else None


def relative_state_expressions(
    phase: Phase,
    third_body_tables: dict[str, ThirdBodyTable],
    layout: StateLayout | None = None,
) -> RelativeStateExpressions:
    """Build the instantaneous RIC state from 12 absolute states and time.

    A mass-carrying relative phase inserts deputy mass before ASSET's time
    column. ``layout`` maps that non-contiguous source into the same compact
    13-argument symbolic expression used by unpowered coupled phases.
    """
    dynamics = phase.dynamics
    model = model_for_phase(phase)
    if (
        dynamics is None
        or model is None
        or model.propagation_mode is not RelativePropagationMode.COUPLED_ECI
    ):
        raise TypeError("Exact RIC expressions require Dynamics.relative(...)")
    arguments = vf.Arguments(13)
    argument_indices = tuple(range(12)) + (12 if layout is None else layout.time_column,)
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
        """Rotate one symbolic inertial vector into instantaneous RIC."""
        return vf.stack(
            [
                radial.dot(vector),
                in_track.dot(vector),
                cross_track.dot(vector),
            ]
        )

    relative_position = rotate_to_ric(deputy_position - chief_position)
    inertial_velocity_difference_ric = rotate_to_ric(deputy_velocity - chief_velocity)
    perturbations = phase_perturbations(phase)
    chief_spacecraft = model.chief_spacecraft
    atmosphere = perturbations.atmosphere
    if perturbations.drag and atmosphere is None:
        atmosphere = EARTH_EXPONENTIAL_ATMOSPHERE
    chief_acceleration = _translational_acceleration(
        chief_position,
        chief_velocity,
        (float(chief_spacecraft.initial_mass_kg) if chief_spacecraft is not None else 1.0),
        mu_m3ps2=float(dynamics.mu_m3ps2),
        include_j2=bool(perturbations.j2),
        central_body_radius_m=float(dynamics.central_body_radius_m),
        j2_coefficient=float(dynamics.j2_coefficient),
        time_var=time,
        third_body_tables=tables_for_phase(phase, third_body_tables),
        include_drag=bool(perturbations.drag),
        include_srp=bool(perturbations.srp),
        cannonball=(chief_spacecraft.cannonball if chief_spacecraft is not None else None),
        atmosphere=atmosphere,
        sun_table=sun_table_for_phase(phase, third_body_tables),
        solar_pressure_at_1au_Npm2=perturbations.solar_pressure_at_1au_Npm2,
    )
    frame_rate_ric = vf.stack(
        [
            chief_position.norm() * cross_track.dot(chief_acceleration) / angular_momentum.norm(),
            chief_position[0] * 0.0,
            angular_momentum.norm() / chief_position.dot(chief_position),
        ]
    )
    relative_velocity = inertial_velocity_difference_ric - frame_rate_ric.cross(relative_position)
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
        argument_indices=argument_indices,
    )


def fix_initial_chief(asset_phase: Any, model: NonlinearRelative) -> None:
    """Fix a propagated chief state at the front of a coupled formulation."""
    if model.propagation_mode not in {
        RelativePropagationMode.COUPLED_ECI,
        RelativePropagationMode.COUPLED_RIC,
    }:
        return
    chief = model.chief_initial_state_eci
    asset_phase.addBoundaryValue(
        "Front",
        ["ChiefR", "ChiefV"],
        np.hstack([chief.r_m, chief.v_mps]),
    )


def apply_geometry_constraint(
    asset_phase: Any,
    constraint: RelativeGeometryConstraint,
    expressions: RelativeStateExpressions,
    *,
    solar_position_table: Any | None = None,
) -> None:
    """Compatibility wrapper for exact-relative geometry ``apply`` methods."""
    constraint.apply(
        asset_phase,
        ConstraintContext(
            vector_functions=vf,
            relative_expressions=expressions,
            is_relative_phase=True,
            solar_position_table=solar_position_table,
        ),
    )


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
        list(expressions.argument_indices),
        [],
        [],
        AutoScale=1.0 / float(velocity_unit_mps),
    )


def _coupled_time_column(raw_trajectory: np.ndarray) -> int:
    """Return the time column for coupled ECI rows with or without mass."""
    raw = np.asarray(raw_trajectory, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 13:
        raise ValueError("Coupled relative trajectory must contain at least 13 columns")
    return 12 if raw.shape[1] == 13 else 13


def coupled_trajectory_rvt(
    raw_trajectory: np.ndarray,
    phase: Phase,
    third_body_tables: dict[str, ThirdBodyTable],
) -> np.ndarray:
    """Convert ``[chief ECI, deputy ECI, t]`` rows to public RIC rows."""
    raw = np.asarray(raw_trajectory, dtype=float)
    time_column = _coupled_time_column(raw)
    dynamics = phase.dynamics
    if dynamics is None or model_for_phase(phase) is None:
        raise TypeError("Coupled extraction requires Dynamics.relative(...)")
    perturbations = phase_perturbations(phase)
    body_tables = tables_for_phase(phase, third_body_tables)
    sun_table = sun_table_for_phase(phase, third_body_tables)
    model = model_for_phase(phase)
    if model is None:
        raise TypeError("Coupled extraction requires Dynamics.relative(...)")
    chief_spacecraft = model.chief_spacecraft
    atmosphere = perturbations.atmosphere
    if perturbations.drag and atmosphere is None:
        atmosphere = EARTH_EXPONENTIAL_ATMOSPHERE
    converted = np.empty((raw.shape[0], 7), dtype=float)
    for index, row in enumerate(raw):
        chief = BoundaryState(row[0:3], row[3:6])
        deputy = BoundaryState(row[6:9], row[9:12])
        chief_acceleration = translational_acceleration_components(
            chief.r_m,
            chief.v_mps,
            time_s=float(row[time_column]),
            mass_kg=(
                float(chief_spacecraft.initial_mass_kg) if chief_spacecraft is not None else 1.0
            ),
            mu_m3ps2=float(dynamics.mu_m3ps2),
            include_j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=body_tables,
            include_drag=bool(perturbations.drag),
            include_srp=bool(perturbations.srp),
            cannonball=(chief_spacecraft.cannonball if chief_spacecraft is not None else None),
            atmosphere=atmosphere,
            sun_table=sun_table,
            solar_pressure_at_1au_Npm2=perturbations.solar_pressure_at_1au_Npm2,
        )
        relative = inertial_to_relative_state(
            chief,
            deputy,
            chief_acceleration_mps2=chief_acceleration,
        )
        converted[index] = np.hstack([relative.r_m, relative.v_mps, float(row[time_column])])
    return converted


def relative_trajectory_rvt(
    raw_trajectory: np.ndarray,
    phase: Phase,
    third_body_tables: dict[str, ThirdBodyTable],
) -> np.ndarray:
    """Return the public RIC view for any nonlinear relative formulation."""
    raw = np.asarray(raw_trajectory, dtype=float)
    model = model_for_phase(phase)
    dynamics = phase.dynamics
    if model is None or dynamics is None:
        raise TypeError("Relative extraction requires Dynamics.relative(...)")
    mode = model.propagation_mode
    if mode is RelativePropagationMode.COUPLED_ECI:
        return coupled_trajectory_rvt(raw, phase, third_body_tables)
    if raw.ndim != 2 or raw.shape[1] < 7:
        raise ValueError("Relative trajectory must contain state and time columns")
    if mode is RelativePropagationMode.COUPLED_RIC:
        if raw.shape[1] < 13:
            raise ValueError("Coupled RIC trajectory must contain 13 columns")
        return np.column_stack([raw[:, 6:12], raw[:, 12]])
    if mode is RelativePropagationMode.NONLINEAR_RIC:
        return raw[:, 0:7].copy()

    converted = np.empty((raw.shape[0], 7), dtype=float)
    for index, row in enumerate(raw):
        chief = propagate_two_body_state(
            model.chief_initial_state_eci,
            float(row[6]),
            float(dynamics.mu_m3ps2),
        )
        relative = (
            relative_orbital_elements_to_relative_state(
                chief,
                row[0:6],
                mu_m3ps2=float(dynamics.mu_m3ps2),
            )
            if mode is RelativePropagationMode.DAMICO
            else classical_relative_orbital_elements_to_relative_state(
                chief,
                row[0:6],
                mu_m3ps2=float(dynamics.mu_m3ps2),
            )
        )
        converted[index] = np.hstack([relative.r_m, relative.v_mps, float(row[6])])
    return converted


def absolute_trajectories(
    raw_trajectory: np.ndarray,
    phase: Phase,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return chief/deputy ECI histories when reconstructable from a mode."""
    raw = np.asarray(raw_trajectory, dtype=float)
    model = model_for_phase(phase)
    dynamics = phase.dynamics
    if model is None or dynamics is None:
        return None, None
    mode = model.propagation_mode
    if mode is RelativePropagationMode.COUPLED_ECI:
        time_column = _coupled_time_column(raw)
        return (
            np.column_stack([raw[:, 0:6], raw[:, time_column]]),
            np.column_stack([raw[:, 6:12], raw[:, time_column]]),
        )

    chief_history = np.empty((raw.shape[0], 7), dtype=float)
    deputy_history = np.empty((raw.shape[0], 7), dtype=float)
    for index, row in enumerate(raw):
        time_index = 12 if mode is RelativePropagationMode.COUPLED_RIC else 6
        time_s = float(row[time_index])
        chief = (
            BoundaryState(row[0:3], row[3:6])
            if mode is RelativePropagationMode.COUPLED_RIC
            else propagate_two_body_state(
                model.chief_initial_state_eci,
                time_s,
                float(dynamics.mu_m3ps2),
            )
        )
        if mode is RelativePropagationMode.COUPLED_RIC:
            relative = BoundaryState(row[6:9], row[9:12])
        elif mode is RelativePropagationMode.NONLINEAR_RIC:
            relative = BoundaryState(row[0:3], row[3:6])
        elif mode is RelativePropagationMode.DAMICO:
            relative = relative_orbital_elements_to_relative_state(
                chief,
                row[0:6],
                mu_m3ps2=float(dynamics.mu_m3ps2),
            )
        else:
            relative = classical_relative_orbital_elements_to_relative_state(
                chief,
                row[0:6],
                mu_m3ps2=float(dynamics.mu_m3ps2),
            )
        deputy = relative_to_inertial_state(chief, relative)
        chief_history[index] = np.hstack([chief.r_m, chief.v_mps, time_s])
        deputy_history[index] = np.hstack([deputy.r_m, deputy.v_mps, time_s])
    return chief_history, deputy_history


def solar_directions_ric(
    raw_trajectory: np.ndarray,
    solar_table: Any,
) -> np.ndarray:
    """Return chief-to-Sun unit directions in instantaneous RIC."""
    raw = np.asarray(raw_trajectory, dtype=float)
    time_column = _coupled_time_column(raw)
    sun_positions = solar_table.sun_position_at(raw[:, time_column])
    directions = np.empty((raw.shape[0], 3), dtype=float)
    for index, (row, sun_position) in enumerate(zip(raw, sun_positions, strict=True)):
        line_eci = sun_position - row[0:3]
        line_ric = ric_basis(row[0:3], row[3:6]) @ line_eci
        directions[index] = line_ric / np.linalg.norm(line_ric)
    return directions
