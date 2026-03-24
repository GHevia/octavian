"""Time helper utilities for composable missions."""

from __future__ import annotations

from collections.abc import Sequence

from .phase import Phase


def cumulative_time_bounds(*dur_bounds: tuple[float, float]) -> list[tuple[float, float]]:
    """Convert per-phase duration bounds into cumulative end-time bounds.

    Args:
        *dur_bounds: Per-phase duration bounds in seconds.

    Returns:
        A list of absolute back-boundary time bounds in seconds, one per phase.

    Example:
        ``cumulative_time_bounds((0, 600), (200, 1000))`` returns
        ``[(0, 600), (200, 1600)]``.
    """
    tmin = 0.0
    tmax = 0.0
    out: list[tuple[float, float]] = []
    for dmin, dmax in dur_bounds:
        tmin += float(dmin)
        tmax += float(dmax)
        out.append((tmin, tmax))
    return out


def normalize_time_bounds(phases: Sequence[Phase]) -> list[tuple[float, float] | None]:
    """Normalize phase time bounds into absolute back-boundary times.

    Args:
        phases: Mission phases in execution order.

    Returns:
        A list of absolute time bounds for each phase. Entries remain ``None``
        for phases without time bounds.

    Notes:
        Phases with ``tof_is_relative=True`` are accumulated from prior phases.
        Phases with absolute bounds reset the running time window.
    """
    out: list[tuple[float, float] | None] = []
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
