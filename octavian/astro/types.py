"""Shared numeric types for astrodynamics helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]


def as_vec3(value: Any) -> Vec3:
    """Convert an input into a length-3 float64 NumPy vector.

    Args:
        value: Array-like object with three elements.

    Returns:
        A ``(3,)`` NumPy array of dtype ``float64``.
    """
    return np.asarray(value, dtype=np.float64).reshape(3)
