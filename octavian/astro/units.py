from __future__ import annotations

from typing import Any

import numpy as np


def default_units(spec: Any) -> tuple[float, float, float]:
    """Choose dimensional units for ASSET auto-scaling.

    Uses boundary magnitudes unless overrides are provided.
    Time scale: average of tf_bounds_s / 10 when available.
    """
    rmag = max(np.linalg.norm(spec.x0.r_m), np.linalg.norm(spec.xf.r_m), 1.0)
    vmag = max(np.linalg.norm(spec.x0.v_mps), np.linalg.norm(spec.xf.v_mps), 1.0)

    if hasattr(spec, "tf_bounds_s"):
        tmin, tmax = map(float, spec.tf_bounds_s)
        tavg = 0.5 * (tmin + tmax)
    else:
        tavg = 10.0
    tmag = max(tavg / 10.0, 1.0)

    r_unit = float(spec.r_unit_m) if getattr(spec, "r_unit_m", None) is not None else float(rmag)
    v_unit = float(spec.v_unit_mps) if getattr(spec, "v_unit_mps", None) is not None else float(vmag)
    t_unit = float(spec.t_unit_s) if getattr(spec, "t_unit_s", None) is not None else float(tmag)
    return r_unit, v_unit, t_unit
