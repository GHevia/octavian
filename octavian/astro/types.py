from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]

def as_vec3(x: Any) -> Vec3:
    return np.asarray(x, dtype=np.float64).reshape(3)
