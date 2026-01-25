from __future__ import annotations

"""Shared numeric types for astrodynamics helpers."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]


def as_vec3(x: Any) -> Vec3:
    """Convert an input to a 3-vector ``float64`` NumPy array.

    Args:
        x: Array-like object with 3 elements.

    Returns:
        A ``(3,)`` array of dtype ``float64``.
    """
    return np.asarray(x, dtype=np.float64).reshape(3)
