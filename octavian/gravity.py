"""Fully normalized spherical-harmonic gravity in a uniformly rotating body frame.

Coefficients use geodetic normalization, without the Condon--Shortley phase:
Pbar_nm = sqrt((2-delta_m0)*(2*n+1)*(n-m)!/(n+m)!) * P_nm.
Only degrees 2 and above are perturbations; central gravity is supplied by the EOM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class SphericalHarmonics:
    """A central body's static, fully normalized gravity coefficients.

    ``cosine[n][m]`` and ``sine[n][m]`` are dimensionless Cbar/Sbar arrays.
    Supply square arrays, including zero entries above the diagonal. C00 may
    be 0 or 1 and is ignored; degree one must be zero (center-of-mass origin).
    For J2 alone, Cbar20 = -J2 / sqrt(5).

    The body rotates about the inertial +Z axis. Its prime-meridian angle is
    ``reference_angle_rad + rotation_rate_radps * (time_s-reference_time_s)``.
    Times are mission-relative seconds, not UTC or SPICE epochs. This simple
    model does not include precession, nutation, polar motion, or tidal changes.
    The coefficient reference radius is independent of the body's mean radius;
    the EOM supplies the gravitational parameter, which must match the field.

    ``backend='python'`` builds ASSET expressions using the same recurrence as
    numerical evaluation. ``backend='cpp'`` uses the optional native extension
    with analytic derivatives, without Python callbacks during integration.
    Degree/order default to the full supplied array. No gravity data is bundled.
    """

    cosine: tuple[tuple[float, ...], ...]
    sine: tuple[tuple[float, ...], ...]
    reference_radius_m: float
    degree: int | None = None
    order: int | None = None
    rotation_rate_radps: float = 0.0
    reference_angle_rad: float = 0.0
    reference_time_s: float = 0.0
    backend: str = "python"

    def __post_init__(self):
        c, s = np.asarray(self.cosine, dtype=float), np.asarray(self.sine, dtype=float)
        if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 3 or s.shape != c.shape:
            raise ValueError("cosine and sine must be matching square arrays of size at least 3")
        if not np.isfinite(c).all() or not np.isfinite(s).all():
            raise ValueError("harmonic coefficients must be finite")
        if np.any(np.triu(c, 1)) or np.any(np.triu(s, 1)) or np.any(s[:, 0]):
            raise ValueError("coefficients above the diagonal and sine[:, 0] must be zero")
        if c[0, 0] not in (0, 1) or np.any(c[1]) or np.any(s[1]):
            raise ValueError("C00 must be 0 or 1 and degree-one coefficients must be zero")
        degree = c.shape[0] - 1 if self.degree is None else self.degree
        order = degree if self.order is None else self.order
        if (
            isinstance(degree, bool)
            or not isinstance(degree, (int, np.integer))
            or not 2 <= degree < c.shape[0]
        ):
            raise ValueError("degree must be an integer from 2 to the supplied maximum degree")
        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or not 0 <= order <= degree
        ):
            raise ValueError("order must be an integer from 0 to degree")
        for name in (
            "reference_radius_m",
            "rotation_rate_radps",
            "reference_angle_rad",
            "reference_time_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (name == "reference_radius_m" and value <= 0):
                raise ValueError(
                    f"{name} must be finite"
                    + (" and positive" if name == "reference_radius_m" else "")
                )
            object.__setattr__(self, name, value)
        if self.backend not in ("python", "cpp"):
            raise ValueError("backend must be 'python' or 'cpp'")
        object.__setattr__(self, "cosine", tuple(tuple(row) for row in c))
        object.__setattr__(self, "sine", tuple(tuple(row) for row in s))
        object.__setattr__(self, "degree", int(degree))
        object.__setattr__(self, "order", int(order))

    @cached_property
    def _native_function(self):
        from ._asset import require_asset

        require_asset("compiled spherical harmonics")
        try:
            import octavian_harmonics_native as native
        except ImportError as exc:
            raise RuntimeError(
                "backend='cpp' requires the optional octavian-harmonics-native extension; "
                "see docs/tutorials/spherical-harmonics.md for build instructions."
            ) from exc
        return native.acceleration_function(self.cosine, self.sine, self.degree, self.order)

    def acceleration(self, position_m, *, mu_m3ps2: float, time_s: float = 0.0):
        """Return inertial perturbing acceleration in m/s² (no point mass)."""
        position = np.asarray(position_m, dtype=float)
        if (
            position.shape != (3,)
            or not np.isfinite(position).all()
            or np.linalg.norm(position) == 0
        ):
            raise ValueError("position_m must contain three finite values with nonzero norm")
        if not math.isfinite(mu_m3ps2) or mu_m3ps2 <= 0 or not math.isfinite(time_s):
            raise ValueError("mu_m3ps2 must be finite and positive and time_s must be finite")
        angle = self.reference_angle_rad + self.rotation_rate_radps * (
            time_s - self.reference_time_s
        )
        c, s = math.cos(angle), math.sin(angle)
        x, y, z = position / self.reference_radius_m
        body = (c * x + s * y, -s * x + c * y, z)
        if self.backend == "cpp":
            ax, ay, az = self._native_function.compute(np.asarray(body))
        else:
            ax, ay, az = _potential_gradient(body, self.cosine, self.sine, self.degree, self.order)
        return (
            mu_m3ps2
            / self.reference_radius_m**2
            * np.asarray((c * ax - s * ay, s * ax + c * ay, az))
        )

    def asset_acceleration(self, position, time, *, mu_m3ps2: float):
        """Compose an ASSET inertial acceleration, including rotation derivatives."""
        from ._asset import require_asset, vf

        require_asset("spherical-harmonic EOMs")
        if time is None:
            if self.rotation_rate_radps:
                raise ValueError("Rotating spherical harmonics require an ASSET time variable")
            time = 0.0
        angle = self.reference_angle_rad + self.rotation_rate_radps * (time - self.reference_time_s)
        c, s = (
            (math.cos(angle), math.sin(angle))
            if isinstance(angle, (int, float))
            else (angle.cos(), angle.sin())
        )
        r = position / self.reference_radius_m
        x, y, z = r[0], r[1], r[2]
        body = (c * x + s * y, -s * x + c * y, z)
        if self.backend == "cpp":
            acc = self._native_function(vf.stack(list(body)))
            ax, ay, az = acc[0], acc[1], acc[2]
        else:
            ax, ay, az = _potential_gradient(body, self.cosine, self.sine, self.degree, self.order)
        # Promote zero components to scalar expressions for empty/truncated fields.
        zero = 0.0 * x
        return (
            mu_m3ps2
            / self.reference_radius_m**2
            * vf.stack([c * ax - s * ay + zero, s * ax + c * ay + zero, az + zero])
        )


def _potential_gradient(position, cosine, sine, degree, order):
    """Acceleration from normalized Cartesian solid harmonics.

    Coordinates are in reference-radius units; V/W include r^(-n-1).
    Cartesian derivatives of degree n use harmonics of degree n+1. These
    identities avoid differentiating the recurrence repeatedly when building
    ASSET expressions, and contain no divisions by distance to the polar axis.
    """
    x, y, z = position
    inverse_r2 = (x * x + y * y + z * z) ** -1.0
    xr, yr, zr = x * inverse_r2, y * inverse_r2, z * inverse_r2
    v, w = {(0, 0): inverse_r2**0.5}, {(0, 0): 0.0}
    for m in range(order + 2):
        if m:
            factor = math.sqrt(3.0) if m == 1 else math.sqrt((2 * m + 1) / (2 * m))
            previous_v, previous_w = v[m - 1, m - 1], w[m - 1, m - 1]
            v[m, m] = factor * (xr * previous_v - yr * previous_w)
            w[m, m] = factor * (xr * previous_w + yr * previous_v)
        for n in range(m + 1, degree + 2):
            a = math.sqrt((4 * n * n - 1) / (n * n - m * m))
            v[n, m] = a * zr * v[n - 1, m]
            w[n, m] = a * zr * w[n - 1, m]
            if n > m + 1:
                b = math.sqrt(
                    (2 * n + 1) * ((n - 1) ** 2 - m * m) / ((2 * n - 3) * (n * n - m * m))
                )
                v[n, m] = v[n, m] - b * inverse_r2 * v[n - 2, m]
                w[n, m] = w[n, m] - b * inverse_r2 * w[n - 2, m]
    ax, ay, az = 0.0, 0.0, 0.0
    for n in range(2, degree + 1):
        for m in range(min(n, order) + 1):
            c, s = cosine[n][m], sine[n][m]
            if c == 0 and s == 0:
                continue
            ratio = (2 * n + 1) / (2 * n + 3)
            vertical = math.sqrt(ratio * ((n + 1) ** 2 - m * m))
            az = az - vertical * (c * v[n + 1, m] + s * w[n + 1, m])
            if m == 0:
                factor = math.sqrt(ratio * (n + 1) * (n + 2) / 2)
                ax = ax - c * factor * v[n + 1, 1]
                ay = ay - c * factor * w[n + 1, 1]
            else:
                upper = math.sqrt(ratio * (n + m + 1) * (n + m + 2))
                lower = math.sqrt(ratio * (n - m + 1) * (n - m + 2) * (2 if m == 1 else 1))
                ax = ax + 0.5 * (
                    -upper * (c * v[n + 1, m + 1] + s * w[n + 1, m + 1])
                    + lower * (c * v[n + 1, m - 1] + s * w[n + 1, m - 1])
                )
                ay = ay + 0.5 * (
                    upper * (-c * w[n + 1, m + 1] + s * v[n + 1, m + 1])
                    + lower * (-c * w[n + 1, m - 1] + s * v[n + 1, m - 1])
                )
    return ax, ay, az
