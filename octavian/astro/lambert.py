from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

try:
    import asset_asrl as ast  # type: ignore
except Exception:  # pragma: no cover
    ast = None  # type: ignore

from .types import Vec3, as_vec3


@dataclass(frozen=True)
class LambertSeed:
    tof_s: float
    longway: bool
    nrev: int
    rightbranch: bool
    v1_mps: Vec3
    v2_mps: Vec3
    total_dv_mps: float


def _call_lambert_izzo(
    r0: Vec3, rf: Vec3, tof_s: float, mu: float, longway: bool, nrev: int, rightbranch: bool
) -> tuple[Vec3, Vec3]:
    if ast is None:
        raise RuntimeError("asset_asrl is required for lambert_izzo.")
    if int(nrev) == 0:
        v1, v2 = ast.Astro.lambert_izzo(r0, rf, float(tof_s), float(mu), bool(longway))
    else:
        v1, v2 = ast.Astro.lambert_izzo(
            r0, rf, float(tof_s), float(mu), bool(longway), int(nrev), bool(rightbranch)
        )
    return as_vec3(v1), as_vec3(v2)


def _total_dv(v1: Vec3, v2: Vec3, v0: Vec3, vf: Vec3) -> float:
    return float(np.linalg.norm(v1 - v0) + np.linalg.norm(vf - v2))


def select_best_lambert_seed(
    *,
    r0_m: Vec3,
    rf_m: Vec3,
    v0_mps: Vec3,
    vf_mps: Vec3,
    mu_m3ps2: float,
    tmin_s: float,
    tmax_s: float,
    n_tofs: int = 50,
    nrevs: Sequence[int] = (0, 1),
) -> LambertSeed:
    """Sweep Lambert solutions and return the seed with minimum total delta-v.

    This helper is designed for *initial guess generation* in impulsive transfers.
    It evaluates a grid of time-of-flight samples and Lambert branch options, then
    chooses the solution that minimizes::

        |v1 - v0| + |vf - v2|

    where ``(v1, v2)`` is the Lambert departure/arrival velocity pair.

    Args:
        r0_m: Initial position [m].
        rf_m: Final position [m].
        v0_mps: Initial velocity [m/s].
        vf_mps: Final velocity [m/s].
        mu_m3ps2: Gravitational parameter [m^3/s^2].
        tmin_s: Minimum time-of-flight to test [s].
        tmax_s: Maximum time-of-flight to test [s].
        n_tofs: Number of TOF samples in ``[tmin_s, tmax_s]``.
        nrevs: Sequence of integer revolution counts to attempt.

    Returns:
        The best Lambert seed found.

    Raises:
        ValueError: If bounds are invalid.
        RuntimeError: If no Lambert solution succeeds over the sweep.
    """
    tmin = float(tmin_s)
    tmax = float(tmax_s)
    if not (tmin > 0.0 and tmax > tmin):
        raise ValueError("Require 0 < tmin_s < tmax_s.")
    if int(n_tofs) < 2:
        raise ValueError("n_tofs must be >= 2.")

    r0 = as_vec3(r0_m)
    rf = as_vec3(rf_m)
    v0 = as_vec3(v0_mps)
    vf = as_vec3(vf_mps)

    best: LambertSeed | None = None
    for tof in np.linspace(tmin, tmax, int(n_tofs)):
        for longway in (False, True):
            for nrev in nrevs:
                rb_iter: Iterable[bool] = (True,) if int(nrev) == 0 else (False, True)
                for rightbranch in rb_iter:
                    try:
                        v1, v2 = _call_lambert_izzo(
                            r0,
                            rf,
                            float(tof),
                            float(mu_m3ps2),
                            bool(longway),
                            int(nrev),
                            bool(rightbranch),
                        )
                    except Exception:
                        continue

                    tot = _total_dv(v1, v2, v0, vf)
                    cand = LambertSeed(
                        tof_s=float(tof),
                        longway=bool(longway),
                        nrev=int(nrev),
                        rightbranch=bool(rightbranch),
                        v1_mps=v1,
                        v2_mps=v2,
                        total_dv_mps=float(tot),
                    )
                    if best is None or cand.total_dv_mps < best.total_dv_mps:
                        best = cand

    if best is None:
        raise RuntimeError("No Lambert solution succeeded for any TOF/branch.")
    return best
