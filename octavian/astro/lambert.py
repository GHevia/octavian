from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple
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

def _call_lambert_izzo(r0: Vec3, rf: Vec3, tof_s: float, mu: float, longway: bool, nrev: int, rightbranch: bool) -> Tuple[Vec3, Vec3]:
    if ast is None:
        raise RuntimeError("asset_asrl is required for lambert_izzo.")
    if int(nrev) == 0:
        v1, v2 = ast.Astro.lambert_izzo(r0, rf, float(tof_s), float(mu), bool(longway))
    else:
        v1, v2 = ast.Astro.lambert_izzo(r0, rf, float(tof_s), float(mu), bool(longway), int(nrev), bool(rightbranch))
    return as_vec3(v1), as_vec3(v2)

def _total_dv(v1: Vec3, v2: Vec3, v0: Vec3, vf: Vec3) -> float:
    return float(np.linalg.norm(v1 - v0) + np.linalg.norm(vf - v2))

def select_best_lambert_seed(*, r0_m: Vec3, rf_m: Vec3, v0_mps: Vec3, vf_mps: Vec3, mu_m3ps2: float,
                             tmin_s: float, tmax_s: float, n_tofs: int = 50, nrevs: Sequence[int] = (0,1)) -> LambertSeed:
    tmin=float(tmin_s); tmax=float(tmax_s)
    if not (tmin>0 and tmax>tmin): raise ValueError("Require 0 < tmin_s < tmax_s")
    if int(n_tofs) < 2: raise ValueError("n_tofs must be >= 2")
    r0=as_vec3(r0_m); rf=as_vec3(rf_m); v0=as_vec3(v0_mps); vf=as_vec3(vf_mps)
    best: Optional[LambertSeed]=None
    for tof in np.linspace(tmin, tmax, int(n_tofs)):
        for longway in (False, True):
            for nrev in nrevs:
                rb_iter: Iterable[bool] = (True,) if int(nrev)==0 else (False, True)
                for rightbranch in rb_iter:
                    try:
                        v1, v2 = _call_lambert_izzo(r0, rf, float(tof), float(mu_m3ps2), bool(longway), int(nrev), bool(rightbranch))
                        tot = _total_dv(v1, v2, v0, vf)
                        cand = LambertSeed(float(tof), bool(longway), int(nrev), bool(rightbranch), v1, v2, float(tot))
                        if best is None or cand.total_dv_mps < best.total_dv_mps:
                            best = cand
                    except Exception:
                        continue
    if best is None:
        raise RuntimeError("No Lambert solution succeeded for any TOF/branch.")
    return best
