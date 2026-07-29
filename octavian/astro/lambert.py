from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .._asset import ast, require_asset
from .types import Vec3, as_vec3

_COLLINEAR_GEOMETRY_TOL = 1.0e-12
_COLLINEAR_ROTATION_RAD = 1.0e-8


@dataclass(frozen=True)
class LambertSeed:
    """One Lambert branch and its endpoint velocity cost."""

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
    """Call ASSET's Lambert solver for one branch."""
    require_asset("Lambert seed generation")
    if int(nrev) == 0:
        v1, v2 = ast.Astro.lambert_izzo(r0, rf, float(tof_s), float(mu), bool(longway))
    else:
        v1, v2 = ast.Astro.lambert_izzo(
            r0, rf, float(tof_s), float(mu), bool(longway), int(nrev), bool(rightbranch)
        )
    return as_vec3(v1), as_vec3(v2)


def _total_dv(v1: Vec3, v2: Vec3, v0: Vec3, vf: Vec3) -> float:
    return float(np.linalg.norm(v1 - v0) + np.linalg.norm(vf - v2))


def _lambert_target_position(
    r0: Vec3,
    rf: Vec3,
    v0: Vec3,
    vf: Vec3,
) -> Vec3:
    """Return a numerically safe Lambert target for collinear endpoints.

    Lambert's transfer plane is undefined when the endpoint position vectors
    are exactly parallel or antiparallel. The mission's boundary constraint is
    still applied to the original target; this function only rotates the seed
    target by a tiny angle in the plane suggested by the endpoint velocities.
    """
    r0_norm = float(np.linalg.norm(r0))
    rf_norm = float(np.linalg.norm(rf))
    if r0_norm <= 0.0 or rf_norm <= 0.0:
        raise ValueError("Lambert endpoint position vectors must be nonzero.")

    sine_of_angle = float(np.linalg.norm(np.cross(r0, rf)) / (r0_norm * rf_norm))
    if sine_of_angle > _COLLINEAR_GEOMETRY_TOL:
        return rf

    plane_normal = _preferred_transfer_plane_normal(r0, rf, v0, vf)
    angle = _COLLINEAR_ROTATION_RAD
    rotated = (
        rf * np.cos(angle)
        + np.cross(plane_normal, rf) * np.sin(angle)
        + plane_normal * np.dot(plane_normal, rf) * (1.0 - np.cos(angle))
    )
    return as_vec3(rotated)


def _preferred_transfer_plane_normal(
    r0: Vec3,
    rf: Vec3,
    v0: Vec3,
    vf: Vec3,
) -> Vec3:
    """Choose a deterministic transfer-plane normal from boundary motion."""
    for candidate in (np.cross(r0, v0), np.cross(rf, vf)):
        magnitude = float(np.linalg.norm(candidate))
        if magnitude > 0.0:
            return as_vec3(candidate / magnitude)

    # Radial or stationary boundary states do not identify a plane. Choose the
    # Cartesian axis least aligned with r0, then form a stable perpendicular.
    r0_direction = r0 / np.linalg.norm(r0)
    basis = np.eye(3)[int(np.argmin(np.abs(r0_direction)))]
    normal = np.cross(r0_direction, basis)
    return as_vec3(normal / np.linalg.norm(normal))


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

    Notes:
        Exactly parallel or antiparallel positions do not define a unique
        Lambert plane. For seed generation only, Octavian rotates the target
        by a tiny deterministic angle in the plane suggested by the boundary
        velocities. The optimization boundary remains the original position.
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
    lambert_rf = _lambert_target_position(r0, rf, v0, vf)

    best: LambertSeed | None = None
    for tof in np.linspace(tmin, tmax, int(n_tofs)):
        for longway in (False, True):
            for nrev in nrevs:
                rb_iter: Iterable[bool] = (True,) if int(nrev) == 0 else (False, True)
                for rightbranch in rb_iter:
                    try:
                        v1, v2 = _call_lambert_izzo(
                            r0,
                            lambert_rf,
                            float(tof),
                            float(mu_m3ps2),
                            bool(longway),
                            int(nrev),
                            bool(rightbranch),
                        )
                    except Exception:
                        continue

                    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
                        # Some Lambert backends return NaNs instead of raising
                        # for singular geometries such as exactly antipodal
                        # endpoints. A non-finite candidate is not a seed.
                        continue
                    tot = _total_dv(v1, v2, v0, vf)
                    if not np.isfinite(tot):
                        continue
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
        raise RuntimeError("No finite Lambert solution succeeded for any TOF/branch.")
    return best
