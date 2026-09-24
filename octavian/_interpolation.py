"""Time-boundary handling for sampled ephemeris reporting."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bounded_query_times(
    query_times_s: ArrayLike,
    sample_times_s: ArrayLike,
    *,
    quantity: str,
) -> NDArray[np.float64]:
    """Clip endpoint roundoff up to 1e-8 s, rejecting extrapolation beyond it.

    Optimized boundary times can differ slightly from their equality targets.
    This tolerance keeps such results reportable without silently accepting
    queries outside the ephemeris coverage. Scalar/array shapes are preserved.
    """
    query = np.asarray(query_times_s, dtype=float)
    samples = np.asarray(sample_times_s, dtype=float)
    if not np.all(np.isfinite(query)):
        raise ValueError("Ephemeris query times must be finite")
    tolerance_s = 1.0e-8
    if np.any(query < samples[0] - tolerance_s) or np.any(query > samples[-1] + tolerance_s):
        raise ValueError(f"Requested {quantity} lies outside the sampled time range")
    return np.clip(query, samples[0], samples[-1])
