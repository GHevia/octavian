from __future__ import annotations

"""Time helper utilities for composable missions."""

from typing import Iterable, List, Optional, Sequence, Tuple

from .phase import Phase


def cumulative_time_bounds(*dur_bounds: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Convert per-phase duration bounds into absolute Back-time bounds.

    Example:
        cumulative_time_bounds((0, 600), (200, 1000)) -> [(0, 600), (200, 1600)]
    """
    tmin = 0.0
    tmax = 0.0
    out: List[Tuple[float, float]] = []
    for dmin, dmax in dur_bounds:
        tmin += float(dmin)
        tmax += float(dmax)
        out.append((tmin, tmax))
    return out


def normalize_time_bounds(phases: Sequence[Phase]) -> List[Optional[Tuple[float, float]]]:
    """Return absolute Back-time bounds for each phase.

    If a phase has tof_is_relative=True, its bounds are treated as per-phase durations
    and accumulated from the previous phases.
    """
    out: List[Optional[Tuple[float, float]]] = []
    tmin = 0.0
    tmax = 0.0
    for ph in phases:
        bounds = ph.tof_bounds_s
        if bounds is None:
            out.append(None)
            continue

        a, b = map(float, bounds)
        if ph.tof_is_relative:
            tmin += a
            tmax += b
            out.append((tmin, tmax))
        else:
            tmin, tmax = a, b
            out.append((tmin, tmax))
    return out
