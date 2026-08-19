"""Time-varying environment tables used by relative-motion phases."""

from __future__ import annotations

from collections.abc import Sequence

from ..data.ephemeris import DEFAULT_EPHEMERIS_BSP
from ..phase import Phase
from ..relative import (
    ClohessyWiltshire,
    NonlinearRelative,
    SolarDirectionTable,
    sample_solar_directions_ric,
)
from .third_bodies import mission_initial_epoch, third_body_table_duration_s


def build_solar_direction_tables(
    mission,
    phases: Sequence[Phase],
    abs_bounds: Sequence[tuple[float, float] | None],
) -> dict[int, SolarDirectionTable]:
    """Build SPICE-derived RIC Sun-direction samples by phase index.

    Every table spans the latest cumulative absolute mission-time bound plus
    the largest configured ``third_body_table_margin_s``. That lets
    multi-phase relative solutions reuse the same SPICE samples for optimizer
    trial points and complete diagnostic histories, even when only one phase
    declares a solar-angle constraint.
    """
    mission_duration_s = third_body_table_duration_s(phases, abs_bounds)
    tables: dict[int, SolarDirectionTable] = {}
    for index, (phase, bounds) in enumerate(zip(phases, abs_bounds, strict=True)):
        solar_constraints = [
            constraint
            for constraint in phase.constraints
            if getattr(constraint, "kind", "") == "solar_phase_angle"
        ]
        if not solar_constraints:
            continue
        dynamics = phase.dynamics
        model = dynamics.model if dynamics is not None else None
        if not isinstance(model, (ClohessyWiltshire, NonlinearRelative)):
            raise ValueError(
                "solar_phase_angle requires Dynamics.cwh(...) or "
                "Dynamics.relative(...) and a RIC phase"
            )
        chief_initial_state = model.chief_initial_state_eci
        if chief_initial_state is None:
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
            chief_initial_state_eci=chief_initial_state,
            mean_motion_radps=model.mean_motion_radps,
            initial_epoch=epoch,
            duration_s=mission_duration_s,
            step_s=float(dynamics.third_body_table_step_s),
            bsp_path=info.get("third_body_bsp_path", DEFAULT_EPHEMERIS_BSP),
        )
    return tables
