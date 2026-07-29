"""Third-body ephemeris table compilation for the composable solver."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .._asset import vf
from ..data.ephemeris import DEFAULT_EPHEMERIS_BSP, sample_sun_moon_positions_eci_tod
from ..dynamics import MOON_MU_M3PS2, SUN_MU_M3PS2, ThirdBodyTable
from ..phase import Phase

if TYPE_CHECKING:  # pragma: no cover
    from ..mission import Mission


SUPPORTED_THIRD_BODY_MU_M3PS2 = {
    "moon": MOON_MU_M3PS2,
    "sun": SUN_MU_M3PS2,
}


def phase_perturbations(phase: Phase):
    """Return supported perturbation flags for a phase.

    This is the composable solver's last validation point before ASSET objects
    are constructed. J2, Moon, and Sun are implemented; SRP, drag, and unknown
    third-body names fail here with an explicit error rather than deeper in the
    vector-function build.
    """
    dynamics = getattr(phase, "dynamics", None)
    if dynamics is None:
        raise ValueError(f"Phase {phase.name!r} is missing dynamics.")
    perturbations = dynamics.active_perturbations()
    unsupported = []
    if perturbations.srp:
        unsupported.append("srp")
    if perturbations.drag:
        unsupported.append("drag")
    third_bodies = perturbations.active_third_bodies()
    unsupported_bodies = [
        body for body in third_bodies if body not in SUPPORTED_THIRD_BODY_MU_M3PS2
    ]
    unsupported.extend(f"third_bodies.{body}" for body in unsupported_bodies)
    if unsupported:
        raise NotImplementedError(
            "Composable solver currently implements J2, Moon, and Sun perturbations only; "
            f"unsupported perturbation flags: {', '.join(unsupported)}."
        )
    return perturbations


def phase_third_body_names(phase: Phase) -> tuple[str, ...]:
    """Return normalized third-body names requested by one phase."""
    return phase_perturbations(phase).active_third_bodies()


def mission_third_body_names(phases: Sequence[Phase]) -> tuple[str, ...]:
    """Return ordered unique third-body names requested anywhere in a mission."""
    names: list[str] = []
    for phase in phases:
        for body in phase_third_body_names(phase):
            if body not in names:
                names.append(body)
    return tuple(names)


def third_body_table_duration_s(
    phases: Sequence[Phase],
    abs_bounds: Sequence[tuple[float, float] | None],
) -> float:
    """Return the mission-relative ephemeris table duration with margin.

    ``abs_bounds`` contains each phase's absolute Back-time bounds after
    ``normalize_time_bounds``. The table must cover the latest possible
    optimizer time, so this uses the maximum finite upper bound across all
    phases. A margin is then added from ``Dynamics.third_body_table_margin_s``;
    when phases disagree, the largest configured margin wins so one shared
    table remains valid for every phase.
    """
    upper_bounds = [float(bounds[1]) for bounds in abs_bounds if bounds is not None]
    if not upper_bounds:
        raise ValueError(
            "Moon/Sun perturbations require finite phase time upper bounds so Octavian "
            "can size the ephemeris interpolation table."
        )
    max_upper_s = max(upper_bounds)
    if not math.isfinite(max_upper_s) or max_upper_s <= 0.0:
        raise ValueError(
            "Moon/Sun ephemeris tables require a finite positive absolute "
            "mission-time upper bound."
        )
    margin_s = max(
        float(getattr(phase.dynamics, "third_body_table_margin_s", 0.0) or 0.0)  # type: ignore[union-attr]
        for phase in phases
        if phase.dynamics is not None
    )
    if not math.isfinite(margin_s) or margin_s < 0.0:
        raise ValueError("third_body_table_margin_s must be finite and non-negative")
    return max_upper_s + margin_s


def mission_initial_epoch(mission: Mission, phases: Sequence[Phase]):
    """Return the epoch used as ephemeris time zero for third-body tables.

    The preferred location is ``Mission.initial_epoch`` because the shared
    interpolation table is mission-relative. ``phases[0].epoch`` is accepted for
    compatibility with the existing phase model, and ``dynamics.info`` remains
    a low-level escape hatch for scripts that build dynamics independently.
    """
    epoch = getattr(mission, "initial_epoch", None)
    if epoch is not None:
        return epoch
    if phases and phases[0].epoch is not None:
        return phases[0].epoch
    for phase in phases:
        dynamics = getattr(phase, "dynamics", None)
        if dynamics is None:
            continue
        info_epoch = getattr(dynamics, "info", {}).get("initial_epoch")
        if info_epoch is not None:
            return info_epoch
    return None


def build_third_body_tables(
    mission: Mission,
    phases: Sequence[Phase],
    abs_bounds: Sequence[tuple[float, float] | None],
) -> dict[str, ThirdBodyTable]:
    """Build shared ASSET interpolation tables for requested third bodies.

    The composable backend uses one time variable per phase, but all phase
    times are mission-relative. This function therefore builds one shared
    table per body covering the whole mission time range. The sample spacing is
    the minimum ``Dynamics.third_body_table_step_s`` among phases, so any phase
    can request a finer table without forcing per-phase ephemeris objects.

    The optional ``dynamics.info["third_body_bsp_path"]`` override is read from
    the first dynamics object to support local experiments with alternate
    kernels. Normal installed-package use relies on ``sun_moon_scheduled.bsp``.
    """
    requested_bodies = mission_third_body_names(phases)
    if not requested_bodies:
        return {}

    initial_epoch = mission_initial_epoch(mission, phases)
    if initial_epoch is None:
        raise ValueError(
            "Moon/Sun perturbations require mission.initial_epoch or the first phase epoch."
        )

    dynamics_with_tables = [phase.dynamics for phase in phases if phase.dynamics is not None]
    step_s = min(float(dyn.third_body_table_step_s) for dyn in dynamics_with_tables)
    duration_s = third_body_table_duration_s(phases, abs_bounds)
    first_info = getattr(dynamics_with_tables[0], "info", {}) if dynamics_with_tables else {}
    times_s, positions_m = sample_sun_moon_positions_eci_tod(
        initial_epoch=initial_epoch,
        duration_s=duration_s,
        step_s=step_s,
        bsp_path=first_info.get("third_body_bsp_path", DEFAULT_EPHEMERIS_BSP),
    )

    tables: dict[str, ThirdBodyTable] = {}
    for body in requested_bodies:
        table = vf.InterpTable1D(times_s, positions_m[body], axis=0, kind="cubic")
        tables[body] = ThirdBodyTable(
            name=body,
            mu_m3ps2=SUPPORTED_THIRD_BODY_MU_M3PS2[body],
            position_table=table,
            times_s=times_s,
            positions_eci_m=positions_m[body],
        )
    return tables


def tables_for_phase(
    phase: Phase,
    third_body_tables: dict[str, ThirdBodyTable],
) -> tuple[ThirdBodyTable, ...]:
    """Return the subset of shared third-body tables requested by ``phase``."""
    return tuple(third_body_tables[name] for name in phase_third_body_names(phase))
