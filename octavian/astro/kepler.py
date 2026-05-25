from __future__ import annotations

import numpy as np

try:
    import asset_asrl as ast  # type: ignore
except Exception:  # pragma: no cover
    ast = None  # type: ignore

from .types import Vec3, as_vec3


def classic_to_cartesian(
    *,
    a_m: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu_m3ps2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert classical orbital elements to Cartesian position and velocity."""
    if ast is not None:
        oe = np.array(
            [
                float(a_m),
                float(e),
                float(np.deg2rad(inc_deg)),
                float(np.deg2rad(raan_deg)),
                float(np.deg2rad(argp_deg)),
                float(np.deg2rad(true_anomaly_deg)),
            ],
            dtype=float,
        )
        rv = np.asarray(ast.Astro.classic_to_cartesian(oe, float(mu_m3ps2)), dtype=float).reshape(6)
        return rv[0:3], rv[3:6]

    inc = np.deg2rad(float(inc_deg))
    raan = np.deg2rad(float(raan_deg))
    argp = np.deg2rad(float(argp_deg))
    nu = np.deg2rad(float(true_anomaly_deg))

    p = float(a_m) * (1.0 - float(e) ** 2)
    r_pf = (p / (1.0 + float(e) * np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0.0], dtype=float)
    v_pf = np.sqrt(float(mu_m3ps2) / p) * np.array(
        [-np.sin(nu), float(e) + np.cos(nu), 0.0],
        dtype=float,
    )

    c_omega = np.cos(raan)
    s_omega = np.sin(raan)
    c_inc = np.cos(inc)
    s_inc = np.sin(inc)
    c_argp = np.cos(argp)
    s_argp = np.sin(argp)
    rot = np.array(
        [
            [c_omega * c_argp - s_omega * s_argp * c_inc, -c_omega * s_argp - s_omega * c_argp * c_inc, s_omega * s_inc],
            [s_omega * c_argp + c_omega * s_argp * c_inc, -s_omega * s_argp + c_omega * c_argp * c_inc, -c_omega * s_inc],
            [s_argp * s_inc, c_argp * s_inc, c_inc],
        ],
        dtype=float,
    )
    return rot @ r_pf, rot @ v_pf


def classical_to_cartesian(
    *,
    a_m: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu_m3ps2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`classic_to_cartesian` using a more explicit name."""
    return classic_to_cartesian(
        a_m=a_m,
        e=e,
        inc_deg=inc_deg,
        raan_deg=raan_deg,
        argp_deg=argp_deg,
        true_anomaly_deg=true_anomaly_deg,
        mu_m3ps2=mu_m3ps2,
    )


def cartesian_to_classic(
    *, r_m: Vec3, v_mps: Vec3, mu_m3ps2: float
) -> dict[str, float]:
    """Convert Cartesian position and velocity to classical orbital elements."""
    r = as_vec3(r_m)
    v = as_vec3(v_mps)
    mu = float(mu_m3ps2)

    if ast is not None:
        rv = np.hstack([r, v]).astype(float)
        oe = np.asarray(ast.Astro.cartesian_to_classic(rv, mu), dtype=float).reshape(6)
        return {
            "a_m": float(oe[0]),
            "e": float(oe[1]),
            "inc_deg": float(np.rad2deg(oe[2])),
            "raan_deg": float(np.rad2deg(oe[3])),
            "argp_deg": float(np.rad2deg(oe[4])),
            "true_anomaly_deg": float(np.rad2deg(oe[5])),
        }

    rnorm = float(np.linalg.norm(r))
    vnorm = float(np.linalg.norm(v))
    if rnorm <= 0.0:
        raise ValueError("r_m must have non-zero norm")

    h = np.cross(r, v)
    hnorm = float(np.linalg.norm(h))
    if hnorm <= 0.0:
        raise ValueError("r_m and v_mps must define a non-degenerate orbit")

    n = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), h)
    nnorm = float(np.linalg.norm(n))
    evec = np.cross(v, h) / mu - r / rnorm
    e = float(np.linalg.norm(evec))
    energy = 0.5 * vnorm**2 - mu / rnorm
    if abs(energy) <= 1e-15:
        raise ValueError("Parabolic orbits are not supported")
    a_m = float(-mu / (2.0 * energy))

    inc = float(np.arccos(np.clip(h[2] / hnorm, -1.0, 1.0)))
    raan = float(np.arctan2(n[1], n[0]) % (2.0 * np.pi)) if nnorm > 0.0 else 0.0

    if e > 0.0 and nnorm > 0.0:
        argp = float(
            np.arctan2(
                np.dot(np.cross(n, evec), h) / (nnorm * hnorm),
                np.dot(n, evec) / nnorm,
            )
            % (2.0 * np.pi)
        )
    else:
        argp = 0.0

    if e > 0.0:
        nu = float(
            np.arctan2(
                np.dot(np.cross(evec, r), h) / (e * hnorm * rnorm),
                np.dot(evec, r) / (e * rnorm),
            )
            % (2.0 * np.pi)
        )
    elif nnorm > 0.0:
        nu = float(
            np.arctan2(
                np.dot(np.cross(n, r), h) / (nnorm * hnorm * rnorm),
                np.dot(n, r) / (nnorm * rnorm),
            )
            % (2.0 * np.pi)
        )
    else:
        nu = float(np.arctan2(r[1], r[0]) % (2.0 * np.pi))

    return {
        "a_m": a_m,
        "e": e,
        "inc_deg": float(np.rad2deg(inc)),
        "raan_deg": float(np.rad2deg(raan)),
        "argp_deg": float(np.rad2deg(argp)),
        "true_anomaly_deg": float(np.rad2deg(nu)),
    }


def propagate_cartesian_rv(rv6: np.ndarray, dt_s: float, mu_m3ps2: float) -> np.ndarray:
    """Propagate a 6D Cartesian state under two-body dynamics using ASSET."""
    if ast is None:
        raise RuntimeError("asset_asrl is required for Kepler propagation.")
    rv6 = np.asarray(rv6, dtype=float).reshape(6)
    out = ast.Astro.propagate_cartesian(rv6, float(dt_s), float(mu_m3ps2))
    return np.asarray(out, dtype=float).reshape(6)


def kepler_dense_guess(
    *, r0_m: Vec3, v0_mps: Vec3, t0_s: float, tf_s: float, npts: int, mu_m3ps2: float
) -> list[np.ndarray]:
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
