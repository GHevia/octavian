"""Time-varying environment tables used by relative-motion phases."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..constraints import SolarPhaseAngle
from ..data.ephemeris import DEFAULT_EPHEMERIS_BSP
from ..phase import Phase
from ..relative import (
    ClohessyWiltshire,
    SolarDirectionTable,
    sample_solar_directions_ric,
)
from ..relative.solar import circular_chief_state
from ..relative.transforms import ric_basis
from .third_bodies import mission_initial_epoch


@dataclass(frozen=True, slots=True)
class RelativeReferenceSamples:
    """Sampled circular-chief position and inertial-to-RIC orientation."""

    times_s: NDArray[np.float64]
    chief_positions_eci_m: NDArray[np.float64]
    inertial_to_ric: NDArray[np.float64]


def build_solar_direction_tables(
    mission,
    phases: Sequence[Phase],
    abs_bounds: Sequence[tuple[float, float] | None],
) -> dict[int, SolarDirectionTable]:
    """Build SPICE-derived RIC Sun-direction samples by phase index."""
    tables: dict[int, SolarDirectionTable] = {}
    for index, (phase, bounds) in enumerate(zip(phases, abs_bounds, strict=True)):
        solar_constraints = [
            constraint
            for constraint in phase.constraints
            if isinstance(constraint, SolarPhaseAngle)
        ]
        if not solar_constraints:
            continue
        dynamics = phase.dynamics
        model = dynamics.model if dynamics is not None else None
        if not isinstance(model, ClohessyWiltshire):
            raise ValueError(
                "solar_phase_angle currently requires Dynamics.cwh(...) and a RIC phase"
            )
        if model.chief_initial_state_eci is None:
            raise ValueError(
                "solar_phase_angle requires Dynamics.cwh("
                "chief_initial_state_eci=...) so the SPICE Sun line can be rotated to RIC"
            )
        epoch = mission_initial_epoch(mission, phases)
        if epoch is None:
            raise ValueError(
                "solar_phase_angle requires Mission.initial_epoch or the first phase epoch"
            )
        if bounds is None or float(bounds[1]) <= 0.0:
            raise ValueError(
                "solar_phase_angle requires a finite positive phase-time upper bound"
            )
        info = dynamics.info if dynamics is not None else {}
        tables[index] = sample_solar_directions_ric(
            chief_initial_state_eci=model.chief_initial_state_eci,
            mean_motion_radps=model.mean_motion_radps,
            initial_epoch=epoch,
            duration_s=float(bounds[1]),
            step_s=float(dynamics.third_body_table_step_s),
            bsp_path=info.get("third_body_bsp_path", DEFAULT_EPHEMERIS_BSP),
        )
    return tables


def build_relative_reference_samples(
    phases: Sequence[Phase],
    abs_bounds: Sequence[tuple[float, float] | None],
) -> dict[int, RelativeReferenceSamples]:
    """Build prescribed chief tables for differentially perturbed CWH phases."""
    samples: dict[int, RelativeReferenceSamples] = {}
    for index, (phase, bounds) in enumerate(zip(phases, abs_bounds, strict=True)):
        dynamics = phase.dynamics
        model = dynamics.model if dynamics is not None else None
        if not isinstance(model, ClohessyWiltshire) or dynamics is None:
            continue
        perturbations = dynamics.active_perturbations()
        has_supported_perturbation = bool(
            perturbations.j2 or perturbations.active_third_bodies()
        )
        if not has_supported_perturbation:
            continue
        if model.chief_initial_state_eci is None:
            raise ValueError(
                "Perturbed CWH dynamics require Dynamics.cwh("
                "chief_initial_state_eci=...) to define the inertial chief reference"
            )
        if bounds is None or float(bounds[1]) <= 0.0:
            raise ValueError(
                "Perturbed CWH dynamics require a finite positive phase-time upper bound"
            )
        duration_s = float(bounds[1])
        requested_step_s = float(dynamics.third_body_table_step_s)
        intervals = max(5, int(np.ceil(duration_s / requested_step_s)))
        times = np.linspace(0.0, duration_s, intervals + 1)
        positions = np.empty((times.size, 3), dtype=float)
        bases = np.empty((times.size, 9), dtype=float)
        for row, time_s in enumerate(times):
            chief = circular_chief_state(
                model.chief_initial_state_eci,
                float(time_s),
                model.mean_motion_radps,
            )
            positions[row] = chief.r_m
            bases[row] = ric_basis(chief.r_m, chief.v_mps).reshape(9)
        samples[index] = RelativeReferenceSamples(
            times_s=times,
            chief_positions_eci_m=positions,
            inertial_to_ric=bases,
        )
    return samples
