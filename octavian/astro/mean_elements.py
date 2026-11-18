"""First-order secular J2 propagation with zero-mean short-period transforms.

The averaging variable is unperturbed mean longitude. A Fourier primitive of
the oscillating Gauss rates defines a near-identity mean-to-osculating map;
its six corrections have zero mean. Mean-longitude corrections include the
change in Keplerian mean motion caused by the periodic shape corrections.
These are not SGP4/TLE or Brouwer-Lyddane elements. Neglected terms are O(J2²).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bodies import EARTH, CelestialBody
from ..specs import BoundaryState
from .equinoctial import (
    equinoctial_elements,
    equinoctial_to_cartesian,
    gauss_matrix,
    mean_longitude,
    true_longitude,
)


@dataclass(frozen=True, slots=True)
class MeanElements:
    """Mean [p,f,g,h,k,lambda], with p in meters and unwrapped lambda in radians.

    f=e cos(Ω+ω), g=e sin(Ω+ω), h=tan(i/2)cosΩ, k=tan(i/2)sinΩ,
    lambda=Ω+ω+M. Circular/equatorial states are regular; the prograde
    equinoctial chart excludes exactly retrograde-equatorial states.
    Use the same central body and J2 setting for both transform directions.
    """

    p_m: float
    f: float
    g: float
    h: float
    k: float
    mean_longitude_rad: float

    def __post_init__(self) -> None:
        values = self.as_array()
        if not np.all(np.isfinite(values)) or self.p_m <= 0 or self.f**2 + self.g**2 >= 1:
            raise ValueError("Mean elements require finite values, positive p, and e < 1.")

    def as_array(self) -> np.ndarray:
        """Return [p,f,g,h,k,lambda] without wrapping longitude."""
        return np.array(
            [self.p_m, self.f, self.g, self.h, self.k, self.mean_longitude_rad], dtype=float
        )

    @property
    def a_m(self) -> float:
        """Mean semi-major axis in meters."""
        return self.p_m / (1 - self.f**2 - self.g**2)


@dataclass(frozen=True, slots=True)
class MeanElementHistory:
    """Mean elements and their model, sampled at elapsed seconds from the input epoch."""

    times_s: np.ndarray
    elements: np.ndarray
    central_body: CelestialBody
    j2: bool
    samples: int

    def to_osculating(self) -> np.ndarray:
        """Reconstruct [R,V,t] in the central body's inertial equatorial frame."""
        rows = []
        for values, time in zip(self.elements, self.times_s, strict=True):
            state = mean_to_osculating(
                MeanElements(*values),
                central_body=self.central_body,
                j2=self.j2,
                samples=self.samples,
            )
            rows.append(np.r_[state.r_m, state.v_mps, time])
        return np.asarray(rows)


def _settings(body, j2, samples):
    if not isinstance(body, CelestialBody):
        raise TypeError("central_body must be a CelestialBody.")
    if not isinstance(j2, bool):
        raise TypeError("j2 must be a boolean.")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 32 or samples % 2:
        raise ValueError("samples must be an even integer >= 32.")
    return body.mu_m3ps2, body.mean_radius_m, body.j2_coefficient if j2 else 0.0


def _rates(elements, longitudes, mu, radius, j2):
    """Perturbation rates in [p,f,g,h,k,mean longitude] coordinates."""
    p, f, g, h, k = elements
    rv = equinoctial_to_cartesian(elements, longitudes, mu)
    r, v = rv[:, :3], rv[:, 3:]
    distances = np.linalg.norm(r, axis=1)
    z2 = (r[:, 2] / distances) ** 2
    acceleration = (
        1.5
        * j2
        * mu
        * radius**2
        / distances[:, None] ** 5
        * r
        * np.column_stack([5 * z2 - 1, 5 * z2 - 1, 5 * z2 - 3])
    )
    radial = r / distances[:, None]
    normal = np.cross(r[0], v[0])
    normal /= np.linalg.norm(normal)
    along = np.cross(normal, radial)
    ric = np.column_stack(
        [
            np.sum(acceleration * radial, axis=1),
            np.sum(acceleration * along, axis=1),
            acceleration @ normal,
        ]
    )
    rates = np.einsum("nij,nj->ni", gauss_matrix(elements, longitudes, mu), ric)
    c, s = np.cos(longitudes), np.sin(longitudes)
    e = np.hypot(f, g)
    w = 1 + f * c + g * s
    dl = (1 - e * e) ** 1.5 / w**2
    if e < 1e-7:
        df, dg = -2 * s, 2 * c
    else:
        anomaly = longitudes - np.arctan2(g, f)
        eccentric = np.arctan2(np.sqrt(1 - e * e) * np.sin(anomaly), e + np.cos(anomaly))
        de = -np.sin(eccentric) * (2 - e * np.cos(eccentric) - e * e) / (1 - e * e)
        df = de * f / e - (1 - dl) * g / e**2
        dg = de * g / e + (1 - dl) * f / e**2
    longitude_rate = (
        dl * np.sqrt(p / mu) * (h * s - k * c) / w * ric[:, 2] + df * rates[:, 1] + dg * rates[:, 2]
    )
    return np.column_stack([rates, longitude_rate])


def _periodic_correction(values, mu, radius, j2, samples):
    if j2 == 0:
        return np.zeros(6)
    p, f, g = values[:3]
    e2 = f * f + g * g
    if p <= 0 or e2 >= 1:
        raise ValueError("Mean/osculating conversion left the elliptic domain.")
    if abs(j2) * (radius * (1 + np.sqrt(e2)) / p) ** 2 > 0.05:
        raise ValueError("J2 perturbation is too large for this first-order mean model.")
    n = np.sqrt(mu / (p / (1 - e2)) ** 3)
    angles = np.arange(samples) * (2 * np.pi / samples)
    longitudes = np.array([true_longitude(values[:5], angle) for angle in angles])
    rates = _rates(values[:5], longitudes, mu, radius, j2)
    modes = np.fft.fftfreq(samples, d=1 / samples)
    primitive = np.zeros((samples, 6), dtype=complex)
    coefficients = np.fft.fft(rates, axis=0) / samples
    primitive[1:, :5] = coefficients[1:, :5] / (1j * modes[1:, None] * n)
    shape_corrections = np.fft.ifft(primitive[:, :5] * samples, axis=0).real
    dn = np.array([-1.5 * n / p, -3 * n * f / (1 - e2), -3 * n * g / (1 - e2), 0, 0])
    phase_source = rates[:, 5] + shape_corrections @ dn
    phase_coefficients = np.fft.fft(phase_source) / samples
    primitive[1:, 5] = phase_coefficients[1:] / (1j * modes[1:] * n)
    return (np.exp(1j * modes * values[5]) @ primitive).real


def osculating_to_mean(
    state: BoundaryState,
    *,
    central_body: CelestialBody = EARTH,
    j2: bool = True,
    samples: int = 128,
) -> MeanElements:
    """Invert the first-order zero-mean Fourier J2 map by fixed-point iteration.

    Inputs are central-body inertial Cartesian coordinates in SI units. This
    is an approximate dynamical model, not an average of Cartesian positions
    over a numerically propagated orbit. The inverse iteration solves the
    truncated map; it does not restore its neglected second-order physics.
    Cartesian states do not retain revolution counts: the returned longitude
    is near its principal branch. Propagation keeps subsequent revolutions.
    Increase samples and check convergence for eccentric orbits.
    """
    mu, radius, coefficient = _settings(central_body, j2, samples)
    elements, longitude = equinoctial_elements(np.r_[state.r_m, state.v_mps], mu)
    osculating = np.r_[elements, mean_longitude(elements, longitude)]
    osculating[5] = (osculating[5] + np.pi) % (2 * np.pi) - np.pi
    current = osculating.copy()
    scale = np.array([elements[0], 1, 1, 1, 1, 1])
    for _ in range(20):
        updated = osculating - _periodic_correction(current, mu, radius, coefficient, samples)
        if np.max(np.abs((updated - current) / scale)) < 1e-12:
            return MeanElements(*updated)
        current = updated
    raise ValueError(
        "Mean/osculating conversion did not converge; increase samples or use numerical propagation."
    )


def mean_to_osculating(
    elements: MeanElements,
    *,
    central_body: CelestialBody = EARTH,
    j2: bool = True,
    samples: int = 128,
) -> BoundaryState:
    """Add the same model's short-period corrections and return Cartesian state."""
    mu, radius, coefficient = _settings(central_body, j2, samples)
    values = elements.as_array()
    corrected = values + _periodic_correction(values, mu, radius, coefficient, samples)
    if corrected[0] <= 0 or corrected[1] ** 2 + corrected[2] ** 2 >= 1:
        raise ValueError("Mean/osculating conversion left the elliptic domain.")
    rv = equinoctial_to_cartesian(corrected[:5], true_longitude(corrected[:5], corrected[5]), mu)
    return BoundaryState(rv[:3], rv[3:])


def propagate_mean_elements(
    initial: MeanElements,
    times_s,
    *,
    central_body: CelestialBody = EARTH,
    j2: bool = True,
    samples: int = 128,
) -> MeanElementHistory:
    """Propagate first-order secular J2 rates; no thrust or other perturbations.

    Times are finite, strictly monotonic seconds relative to the initial mean
    elements, and may be positive or negative. Mean a/e/i remain constant.
    Node, periapsis, and mean anomaly drift at their first-order secular rates.
    """
    mu, radius, coefficient = _settings(central_body, j2, samples)
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)):
        raise ValueError("times_s must be a nonempty finite one-dimensional array.")
    if len(times) > 1 and not (np.all(np.diff(times) > 0) or np.all(np.diff(times) < 0)):
        raise ValueError("times_s must be strictly monotonic.")
    p, f, g, h, k, longitude = initial.as_array()
    n = np.sqrt(mu / initial.a_m**3)
    cos_i = (1 - h * h - k * k) / (1 + h * h + k * k)
    factor = 0.75 * coefficient * n * (radius / p) ** 2
    node_rate = -2 * factor * cos_i
    periapsis_rate = factor * (5 * cos_i * cos_i - 1)
    anomaly_rate = n + factor * np.sqrt(1 - f * f - g * g) * (3 * cos_i * cos_i - 1)
    varpi = (node_rate + periapsis_rate) * times
    node = node_rate * times
    rows = np.column_stack(
        [
            np.full_like(times, p),
            f * np.cos(varpi) - g * np.sin(varpi),
            f * np.sin(varpi) + g * np.cos(varpi),
            h * np.cos(node) - k * np.sin(node),
            h * np.sin(node) + k * np.cos(node),
            longitude + (node_rate + periapsis_rate + anomaly_rate) * times,
        ]
    )
    return MeanElementHistory(times.copy(), rows, central_body, j2, samples)
