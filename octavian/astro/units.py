"""Dimensional scaling helpers for ASSET-backed solves."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..coordinates import SolverScaling


def default_scaling(spec: Any) -> SolverScaling:
    """Choose characteristic units for a dimensional solver problem.

    An explicit ``spec.scaling`` declaration wins. Otherwise the legacy
    endpoint-derived length, velocity, and time rules are preserved, with an
    optional mass scale read from ``mass_unit_kg``.
    """
    configured = getattr(spec, "scaling", None)
    if configured is not None:
        if not isinstance(configured, SolverScaling):
            raise TypeError("scaling must be a SolverScaling instance")
        return configured

    radius_scale_m = max(np.linalg.norm(spec.x0.r_m), np.linalg.norm(spec.xf.r_m), 1.0)
    velocity_scale_mps = max(np.linalg.norm(spec.x0.v_mps), np.linalg.norm(spec.xf.v_mps), 1.0)

    if hasattr(spec, "tf_bounds_s"):
        t_min_s, t_max_s = map(float, spec.tf_bounds_s)
        average_time_s = 0.5 * (t_min_s + t_max_s)
    else:
        average_time_s = 10.0
    time_scale_s = max(average_time_s / 10.0, 1.0)

    return SolverScaling(
        length_m=(
            float(spec.r_unit_m)
            if getattr(spec, "r_unit_m", None) is not None
            else float(radius_scale_m)
        ),
        velocity_mps=(
            float(spec.v_unit_mps)
            if getattr(spec, "v_unit_mps", None) is not None
            else float(velocity_scale_mps)
        ),
        time_s=(
            float(spec.t_unit_s)
            if getattr(spec, "t_unit_s", None) is not None
            else float(time_scale_s)
        ),
        mass_kg=float(getattr(spec, "mass_unit_kg", None) or 1.0),
    )


def default_units(spec: Any) -> tuple[float, float, float]:
    """Choose default dimensional units for solver auto-scaling.

    Args:
        spec: Object with boundary states ``x0`` and ``xf`` and optional unit
            overrides such as ``r_unit_m``, ``v_unit_mps``, and ``t_unit_s``.

    Returns:
        A tuple ``(r_unit_m, v_unit_mps, t_unit_s)`` suitable for ASSET phase
        scaling.
    """
    scaling = default_scaling(spec)
    return scaling.length_m, scaling.velocity_mps, scaling.time_s
