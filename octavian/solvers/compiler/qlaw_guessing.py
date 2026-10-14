"""Numerical equinoctial Q-law seeds, independent of the ASSET backend."""

from __future__ import annotations

import numpy as np

from ...guesses import LowThrustQLawGuess
from .powered_guessing import G0_MPS2


def equinoctial_elements(rv: np.ndarray, mu: float) -> tuple[np.ndarray, float]:
    """Return modified equinoctial [p,f,g,h,k] and true longitude."""
    rv = np.asarray(rv, dtype=float)
    if rv.shape != (6,) or not np.all(np.isfinite(rv)) or not np.isfinite(mu) or mu <= 0:
        raise ValueError("Q-law requires a finite six-component Cartesian state and positive mu.")
    r, v = rv[:3], rv[3:6]
    radius = np.linalg.norm(r)
    momentum = np.cross(r, v)
    magnitude = np.linalg.norm(momentum)
    if radius <= 0 or magnitude <= 0:
        raise ValueError("Q-law requires nonzero radius and angular momentum.")
    normal = momentum / magnitude
    if 1.0 + normal[2] < 1e-6:
        raise ValueError("Q-law does not support the retrograde equinoctial singularity.")
    h, k = -normal[1] / (1 + normal[2]), normal[0] / (1 + normal[2])
    s = 1 + h * h + k * k
    f_basis = np.array([1 + h * h - k * k, 2 * h * k, -2 * k]) / s
    g_basis = np.array([2 * h * k, 1 - h * h + k * k, 2 * h]) / s
    eccentricity = np.cross(v, momentum) / mu - r / radius
    f, g = eccentricity @ f_basis, eccentricity @ g_basis
    if f * f + g * g >= 1:
        raise ValueError("Q-law seeds require bound elliptic orbits.")
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


def qlaw_direction(
    rv: np.ndarray,
    target: np.ndarray,
    mu: float,
    orbit_samples: int,
) -> np.ndarray:
    """Choose unit RIC thrust minimizing the instantaneous quotient derivative.

    Q is the sum of squared element errors divided by the squared maximum
    element rates per unit acceleration over sampled orbital longitudes.
    The numerical gradient includes derivatives of those rate denominators.
    """
    elements, longitude = equinoctial_elements(rv, mu)
    scale = np.array([target[0], 1, 1, 1, 1])
    normalized = elements / scale
    epsilon = 1e-5
    perturbation = np.eye(5) * epsilon
    candidates = np.concatenate([normalized + perturbation, normalized - perturbation])
    physical = candidates * scale
    longitudes = np.linspace(0, 2 * np.pi, orbit_samples, endpoint=False)
    rates = gauss_matrix(physical[:, None, :], longitudes[None, :], mu)
    rates = rates / scale[None, None, :, None]
    maximum_squared = np.max(np.sum(rates * rates, axis=-1), axis=1)
    errors = candidates - target / scale
    quotient = np.sum(errors * errors / np.maximum(maximum_squared, 1e-30), axis=1)
    gradient = (quotient[:5] - quotient[5:]) / (2 * epsilon)
    matrix = gauss_matrix(elements, np.asarray(longitude), mu) / scale[:, None]
    descent = -(gradient @ matrix)
    norm = np.linalg.norm(descent)
    return descent / norm if norm > 1e-20 else np.zeros(3)


def build_low_thrust_qlaw_seed(
    *,
    initial_rv: np.ndarray,
    target_rv: np.ndarray,
    mu_m3ps2: float,
    initial_mass_kg: float,
    dry_mass_kg: float,
    thrust_N: float,
    isp_s: float,
    time_bounds_s: tuple[float, float],
    npts: int,
    config: LowThrustQLawGuess,
) -> tuple[list[np.ndarray], dict[str, float | int | bool | str]]:
    """Integrate two-body Q-law steering into inertial [R,V,M,t,U] rows.

    Stop at target tolerances or the upper duration bound. If the target is
    reached before the lower bound, coast until the lower bound. Exhausting
    propellant before target arrival raises an actionable error.
    """
    mu = float(mu_m3ps2)
    tmin, tmax = map(float, time_bounds_s)
    if not (mu > 0 and thrust_N > 0 and isp_s > 0 and initial_mass_kg > dry_mass_kg > 0):
        raise ValueError("Q-law requires positive dynamics/propulsion and usable propellant.")
    if not 0 <= tmin < tmax:
        raise ValueError("Q-law time bounds must satisfy max > min >= 0.")
    target, _ = equinoctial_elements(np.asarray(target_rv), mu)
    target_a = target[0] / (1 - target[1] ** 2 - target[2] ** 2)
    target_normal = np.cross(target_rv[:3], target_rv[3:6]).astype(float)
    target_normal /= np.linalg.norm(target_normal)
    target_evec = np.cross(target_rv[3:6], np.cross(target_rv[:3], target_rv[3:6])) / mu
    target_evec -= target_rv[:3] / np.linalg.norm(target_rv[:3])
    state = np.r_[initial_rv, initial_mass_kg].astype(float)
    equinoctial_elements(state[:6], mu)
    coast = False

    def errors(current: np.ndarray) -> tuple[float, float, float]:
        r, v = current[:3], current[3:6]
        momentum = np.cross(r, v)
        normal = momentum / np.linalg.norm(momentum)
        a = 1 / (2 / np.linalg.norm(r) - v @ v / mu)
        evec = np.cross(v, momentum) / mu - r / np.linalg.norm(r)
        angle = np.arctan2(np.linalg.norm(np.cross(normal, target_normal)), normal @ target_normal)
        return abs(a / target_a - 1), np.linalg.norm(evec - target_evec), float(np.rad2deg(angle))

    def control(current: np.ndarray) -> np.ndarray:
        if coast:
            return np.zeros(3)
        r, v = current[:3], current[3:6]
        radial = r / np.linalg.norm(r)
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal)
        basis = np.column_stack([radial, np.cross(normal, radial), normal])
        return (
            config.throttle * basis @ qlaw_direction(current[:6], target, mu, config.orbit_samples)
        )

    def derivative(current: np.ndarray) -> np.ndarray:
        u = control(current)
        r = current[:3]
        return np.r_[
            current[3:6],
            -mu * r / np.linalg.norm(r) ** 3 + thrust_N * u / current[6],
            -thrust_N * np.linalg.norm(u) / (isp_s * G0_MPS2),
        ]

    time = 0.0
    history = [np.r_[state, time, control(state)]]
    integration_steps = 0
    reached = False
    while time < tmax:
        if coast and time >= tmin:
            break
        a_error, e_error, plane_error = errors(state)
        reached = (
            a_error <= config.semi_major_axis_rtol
            and e_error <= config.eccentricity_tol
            and plane_error <= config.plane_tol_deg
        )
        if reached:
            if time >= tmin:
                break
            coast = True
            history[-1][8:11] = 0.0
        elements, _ = equinoctial_elements(state[:6], mu)
        a = elements[0] / (1 - elements[1] ** 2 - elements[2] ** 2)
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        # Also resolve the fastest motion near periapsis in eccentric orbits.
        eccentricity = np.hypot(elements[1], elements[2])
        step = min(period * (1 - eccentricity) ** 1.5 / config.steps_per_orbit, tmax - time)
        if coast:
            step = min(step, tmin - time)
        consumption = thrust_N * config.throttle * step / (isp_s * G0_MPS2)
        if not coast and state[6] - consumption <= dry_mass_kg:
            raise ValueError(
                "Q-law seed exhausts usable propellant before reaching the target; "
                "add propellant or reduce the requested orbit change."
            )
        k1 = derivative(state)
        k2 = derivative(state + step * k1 / 2)
        k3 = derivative(state + step * k2 / 2)
        k4 = derivative(state + step * k3)
        state += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        time += step
        history.append(np.r_[state, time, control(state)])
        integration_steps += 1
    if time <= 0:
        raise ValueError("Q-law target already matches the initial orbit; use a coast guess.")
    dense = np.asarray(history)
    times = np.linspace(0, time, max(npts, 2))
    # Preserve the integrated orbital geometry when resampling for ASSET.
    sampled = np.column_stack([np.interp(times, dense[:, 7], dense[:, j]) for j in range(7)])
    controls = np.column_stack([np.interp(times, dense[:, 7], dense[:, j]) for j in range(8, 11)])
    rows = [np.r_[row, t, u] for row, t, u in zip(sampled, times, controls, strict=True)]
    a_error, e_error, plane_error = errors(state)
    reached = (
        a_error <= config.semi_major_axis_rtol
        and e_error <= config.eccentricity_tol
        and plane_error <= config.plane_tol_deg
    )
    return rows, {
        "guess_kind": "low_thrust_qlaw",
        "seed_tof_s": float(time),
        "seed_throttle": config.throttle,
        "seed_final_mass_kg": float(state[6]),
        "seed_integration_steps": integration_steps,
        "seed_target_reached": bool(reached),
        "seed_semi_major_axis_relative_error": float(a_error),
        "seed_eccentricity_vector_error": float(e_error),
        "seed_plane_error_deg": plane_error,
    }
