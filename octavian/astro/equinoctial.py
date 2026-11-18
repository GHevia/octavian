"""Modified equinoctial conversions and Gauss variational equations."""

from __future__ import annotations

import numpy as np


def equinoctial_elements(rv: np.ndarray, mu: float) -> tuple[np.ndarray, float]:
    """Return modified equinoctial [p,f,g,h,k] and true longitude."""
    rv = np.asarray(rv, dtype=float)
    if rv.shape != (6,) or not np.all(np.isfinite(rv)) or not np.isfinite(mu) or mu <= 0:
        raise ValueError(
            "Equinoctial conversion requires a finite six-component state and positive mu."
        )
    r, v = rv[:3], rv[3:6]
    radius = np.linalg.norm(r)
    momentum = np.cross(r, v)
    magnitude = np.linalg.norm(momentum)
    if radius <= 0 or magnitude <= 0:
        raise ValueError("Equinoctial conversion requires nonzero radius and angular momentum.")
    normal = momentum / magnitude
    if 1.0 + normal[2] < 1e-6:
        raise ValueError("The prograde equinoctial chart excludes the retrograde singularity.")
    h, k = -normal[1] / (1 + normal[2]), normal[0] / (1 + normal[2])
    s = 1 + h * h + k * k
    f_basis = np.array([1 + h * h - k * k, 2 * h * k, -2 * k]) / s
    g_basis = np.array([2 * h * k, 1 - h * h + k * k, 2 * h]) / s
    eccentricity = np.cross(v, momentum) / mu - r / radius
    f, g = eccentricity @ f_basis, eccentricity @ g_basis
    if f * f + g * g >= 1:
        raise ValueError("Equinoctial conversion requires a bound elliptic orbit.")
    return np.array([magnitude * magnitude / mu, f, g, h, k]), float(
        np.arctan2(r @ g_basis, r @ f_basis)
    )


def gauss_matrix(elements: np.ndarray, longitude: np.ndarray, mu: float) -> np.ndarray:
    """Map RIC acceleration to equinoctial element rates; supports batching."""
    p, f, g, h, k = np.moveaxis(elements, -1, 0)
    c, s = np.cos(longitude), np.sin(longitude)
    w = 1 + f * c + g * s
    q = h * s - k * c
    zero = np.zeros_like(w)
    rows = [
        np.stack([zero, 2 * p / w, zero], axis=-1),
        np.stack([s + zero, ((w + 1) * c + f) / w, -q * g / w], axis=-1),
        np.stack([-c + zero, ((w + 1) * s + g) / w, q * f / w], axis=-1),
        np.stack([zero, zero, (1 + h * h + k * k) * c / (2 * w)], axis=-1),
        np.stack([zero, zero, (1 + h * h + k * k) * s / (2 * w)], axis=-1),
    ]
    return np.stack(rows, axis=-2) * np.sqrt(p / mu)[..., None, None]


def equinoctial_to_cartesian(elements, longitude, mu):
    """Convert fixed [p,f,g,h,k] at one or many true longitudes to Cartesian RV."""
    p, f, g, h, k = elements
    s2 = 1 + h * h + k * k
    first = np.array([1 + h * h - k * k, 2 * h * k, -2 * k]) / s2
    second = np.array([2 * h * k, 1 - h * h + k * k, 2 * h]) / s2
    c, sn = np.cos(longitude), np.sin(longitude)
    r = (p / (1 + f * c + g * sn))[..., None] * (c[..., None] * first + sn[..., None] * second)
    v = np.sqrt(mu / p) * (-(sn + g)[..., None] * first + (c + f)[..., None] * second)
    return np.concatenate([r, v], axis=-1)


def mean_longitude(elements, longitude):
    """Convert true to mean equinoctial longitude."""
    e = np.hypot(elements[1], elements[2])
    if e < 1e-10:
        return longitude
    varpi = np.arctan2(elements[2], elements[1])
    nu = longitude - varpi
    eccentric = np.arctan2(np.sqrt(1 - e * e) * np.sin(nu), e + np.cos(nu))
    return varpi + eccentric - e * np.sin(eccentric)


def true_longitude(elements, mean):
    """Recover true longitude by solving Kepler's equation."""
    e = np.hypot(elements[1], elements[2])
    if e < 1e-10:
        return mean
    varpi = np.arctan2(elements[2], elements[1])
    anomaly = (mean - varpi + np.pi) % (2 * np.pi) - np.pi
    eccentric = anomaly + np.copysign(0.85 * e, np.sin(anomaly))
    for _ in range(20):
        update = (eccentric - e * np.sin(eccentric) - anomaly) / (1 - e * np.cos(eccentric))
        eccentric -= update
        if abs(update) < 1e-12:
            break
    return varpi + np.arctan2(np.sqrt(1 - e * e) * np.sin(eccentric), np.cos(eccentric) - e)
