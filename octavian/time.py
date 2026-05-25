"""Time helper utilities for missions."""

from __future__ import annotations

from collections.abc import Sequence

from .phase import Phase


def cumulative_time_bounds(*duration_bounds_s: tuple[float, float]) -> list[tuple[float, float]]:
    """Convert per-phase duration bounds into cumulative end-time bounds.

    Args:
        *duration_bounds_s: Per-phase duration bounds in seconds.

    Returns:
        Absolute back-boundary time bounds in seconds, one tuple per phase.
    """
    cumulative_min_s = 0.0
    cumulative_max_s = 0.0
    absolute_bounds: list[tuple[float, float]] = []
    for duration_min_s, duration_max_s in duration_bounds_s:
        cumulative_min_s += float(duration_min_s)
        cumulative_max_s += float(duration_max_s)
        absolute_bounds.append((cumulative_min_s, cumulative_max_s))
    return absolute_bounds


def normalize_time_bounds(phases: Sequence[Phase]) -> list[tuple[float, float] | None]:
    """Normalize phase time bounds into absolute back-boundary times.

    Args:
        phases: Mission phases in execution order.

    Returns:
        A list of absolute time bounds for each phase. Entries remain ``None``
        for phases without time bounds.
    """
    normalized_bounds: list[tuple[float, float] | None] = []
    running_min_s = 0.0
    running_max_s = 0.0
    for phase in phases:
        phase_bounds_s = phase.tof_bounds_s
        if phase_bounds_s is None:
            normalized_bounds.append(None)
            continue

        phase_min_s, phase_max_s = map(float, phase_bounds_s)
        if phase.tof_is_relative:
            running_min_s += phase_min_s
            running_max_s += phase_max_s
            normalized_bounds.append((running_min_s, running_max_s))
        else:
            running_min_s, running_max_s = phase_min_s, phase_max_s
            normalized_bounds.append((running_min_s, running_max_s))
    return normalized_bounds
