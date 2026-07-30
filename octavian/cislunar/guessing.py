"""Initial-guess builders for dimensional CR3BP phases."""

from __future__ import annotations

import numpy as np

from ..specs import BoundaryState


def cr3bp_hermite_guess(
    initial_state: BoundaryState,
    final_state: BoundaryState,
    *,
    t0_s: float,
    tf_s: float,
    npts: int,
) -> list[np.ndarray]:
    """Build a boundary-matching cubic Hermite state guess.

    This geometric seed matches both endpoint positions and velocities. It is
    intended for CR3BP collocation initialization; the optimizer enforces the
    actual equations of motion.

    Args:
        initial_state: Dimensional synodic state at ``t0_s``.
        final_state: Dimensional synodic state at ``tf_s``.
        t0_s: Initial phase time.
        tf_s: Final phase time.
        npts: Number of returned rows, including endpoints.
    """
    start = float(t0_s)
    end = float(tf_s)
    if end <= start:
        raise ValueError("CR3BP guess requires tf_s > t0_s")
    if int(npts) < 2:
        raise ValueError("CR3BP guess requires at least two points")
    duration = end - start
    times = np.linspace(start, end, int(npts))
    rows: list[np.ndarray] = []
    for time_s in times:
        fraction = (time_s - start) / duration
        fraction_sq = fraction**2
        fraction_cu = fraction**3
        h00 = 2.0 * fraction_cu - 3.0 * fraction_sq + 1.0
        h10 = fraction_cu - 2.0 * fraction_sq + fraction
        h01 = -2.0 * fraction_cu + 3.0 * fraction_sq
        h11 = fraction_cu - fraction_sq
        position = (
            h00 * initial_state.r_m
            + h10 * duration * initial_state.v_mps
            + h01 * final_state.r_m
            + h11 * duration * final_state.v_mps
        )
        dh00 = (6.0 * fraction_sq - 6.0 * fraction) / duration
        dh10 = 3.0 * fraction_sq - 4.0 * fraction + 1.0
        dh01 = (-6.0 * fraction_sq + 6.0 * fraction) / duration
        dh11 = 3.0 * fraction_sq - 2.0 * fraction
        velocity = (
            dh00 * initial_state.r_m
            + dh10 * initial_state.v_mps
            + dh01 * final_state.r_m
            + dh11 * final_state.v_mps
        )
        rows.append(np.hstack([position, velocity, time_s]))
    return rows
