"""Position validation shared by trajectory visualization backends."""

from __future__ import annotations

import numpy as np


def reference_positions(value: object) -> np.ndarray:
    """Extract finite XYZ columns from a non-empty position or state history."""
    rows = np.asarray(value, dtype=float)
    if (
        rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] < 3
        or not np.all(np.isfinite(rows[:, 0:3]))
    ):
        raise ValueError("Each reference trajectory must contain finite position rows")
    return rows[:, :3]
