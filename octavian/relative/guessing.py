"""Initial-guess generation for impulsive CWH rendezvous phases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..specs import BoundaryState
from .cwh import cwh_rendezvous_velocity, propagate_cwh


@dataclass(frozen=True, slots=True)
class CWHRendezvousSeed:
    """Feasible position-to-position CWH arc and its impulsive cost."""

    tof_s: float
    departure_velocity_mps: np.ndarray
    arrival_velocity_mps: np.ndarray
    total_dv_mps: float


def select_cwh_rendezvous_seed(
    initial_state: BoundaryState,
    final_state: BoundaryState,
    *,
    mean_motion_radps: float,
    tof_bounds_s: tuple[float, float],
    samples: int = 60,
    candidate_filter: Callable[[CWHRendezvousSeed], bool] | None = None,
) -> CWHRendezvousSeed:
    """Select the lowest-impulse nonsingular CWH seed over a time grid.

    When ``candidate_filter`` is supplied, the lowest-cost accepted seed wins.
    If no candidate is accepted, the lowest-cost unfiltered seed is returned so
    the optimizer still receives a deterministic fallback.
    """
    minimum_time_s, maximum_time_s = map(float, tof_bounds_s)
    if not (0.0 < minimum_time_s < maximum_time_s):
        raise ValueError("tof_bounds_s must satisfy 0 < minimum < maximum")
    if int(samples) < 2:
        raise ValueError("samples must be at least 2")

    best: CWHRendezvousSeed | None = None
    best_accepted: CWHRendezvousSeed | None = None
    for tof_s in np.linspace(minimum_time_s, maximum_time_s, int(samples)):
        try:
            departure_velocity = cwh_rendezvous_velocity(
                initial_state.r_m,
                final_state.r_m,
                float(tof_s),
                mean_motion_radps,
            )
        except ValueError:
            continue
        departure_state = np.hstack([initial_state.r_m, departure_velocity])
        arrival_state = propagate_cwh(departure_state, float(tof_s), mean_motion_radps)
        if not np.all(np.isfinite(arrival_state)):
            continue
        total_dv_mps = float(
            np.linalg.norm(departure_velocity - initial_state.v_mps)
            + np.linalg.norm(final_state.v_mps - arrival_state[3:6])
        )
        candidate = CWHRendezvousSeed(
            tof_s=float(tof_s),
            departure_velocity_mps=departure_velocity,
            arrival_velocity_mps=arrival_state[3:6],
            total_dv_mps=total_dv_mps,
        )
        if best is None or candidate.total_dv_mps < best.total_dv_mps:
            best = candidate
        if (
            candidate_filter is not None
            and candidate_filter(candidate)
            and (
                best_accepted is None
                or candidate.total_dv_mps < best_accepted.total_dv_mps
            )
        ):
            best_accepted = candidate

    if best is None:
        raise RuntimeError("No nonsingular CWH rendezvous seed exists in the time bounds")
    return best_accepted or best


def cwh_dense_guess(
    initial_position_m: np.ndarray,
    departure_velocity_mps: np.ndarray,
    *,
    mean_motion_radps: float,
    t0_s: float,
    tf_s: float,
    npts: int,
) -> list[np.ndarray]:
    """Build dense ``[relative state, time]`` rows from an analytic CWH arc."""
    if int(npts) < 2:
        raise ValueError("npts must be at least 2")
    if float(tf_s) <= float(t0_s):
        raise ValueError("tf_s must be later than t0_s")
    initial_state = np.hstack(
        [
            np.asarray(initial_position_m, dtype=float).reshape(3),
            np.asarray(departure_velocity_mps, dtype=float).reshape(3),
        ]
    )
    return [
        np.hstack(
            [
                propagate_cwh(initial_state, float(time_s - t0_s), mean_motion_radps),
                float(time_s),
            ]
        )
        for time_s in np.linspace(float(t0_s), float(tf_s), int(npts))
    ]
