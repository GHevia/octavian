from __future__ import annotations

import numpy as np

try:
    import asset_asrl as ast  # type: ignore
except Exception:  # pragma: no cover
    ast = None  # type: ignore

from .types import Vec3, as_vec3


def propagate_cartesian_rv(rv6: np.ndarray, dt_s: float, mu_m3ps2: float) -> np.ndarray:
    """Propagate a 6D Cartesian state under two-body dynamics using ASSET."""
    if ast is None:
        raise RuntimeError("asset_asrl is required for Kepler propagation.")
    rv6 = np.asarray(rv6, dtype=float).reshape(6)
    out = ast.Astro.propagate_cartesian(rv6, float(dt_s), float(mu_m3ps2))
    return np.asarray(out, dtype=float).reshape(6)

def kepler_dense_guess(*, r0_m: Vec3, v0_mps: Vec3, t0_s: float, tf_s: float, npts: int, mu_m3ps2: float) -> list[np.ndarray]:
    """Dense guess [x(6), t] built via repeated Kepler propagation."""
    if npts < 2:
        raise ValueError("npts must be >= 2")
    t0 = float(t0_s)
    tf = float(tf_s)
    if tf <= t0:
        raise ValueError("Require tf_s > t0_s")
    rv0 = np.hstack([as_vec3(r0_m), as_vec3(v0_mps)])
    ts = np.linspace(t0, tf, int(npts))
    out: list[np.ndarray] = []
    for t in ts:
        rv = propagate_cartesian_rv(rv0, float(t - t0), float(mu_m3ps2))
        out.append(np.hstack([rv, float(t)]))
    return out

def estimate_orbital_period_s(r_m: Vec3, v_mps: Vec3, mu_m3ps2: float) -> float | None:
    """Estimate orbital period for elliptic orbit; else None."""
    r = as_vec3(r_m)
    v = as_vec3(v_mps)
    mu = float(mu_m3ps2)
    rnorm = float(np.linalg.norm(r))
    if rnorm <= 0:
        return None
    eps = 0.5 * float(v @ v) - mu / rnorm
    if eps >= 0:
        return None
    a = -mu / (2 * eps)
    if a <= 0:
        return None
    return float(2 * np.pi * np.sqrt(a**3 / mu))
