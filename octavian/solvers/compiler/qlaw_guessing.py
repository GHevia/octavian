"""Numerical equinoctial Q-law seeds, independent of the ASSET backend."""

from __future__ import annotations

import numpy as np

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


def proximity_quotient_gradient(
    elements: np.ndarray,
    target: np.ndarray,
    mu: float,
    orbit_samples: int,
) -> tuple[float, np.ndarray]:
    """Return the target-normalized proximity quotient and analytic gradient.

    Reference rates are fixed at the target orbit: increasing the current
    orbit's achievable rates must not make an unchanged error look smaller.
    Plane error uses the angle between normals, rather than distance in the
    stereographic (h,k) chart. This also avoids steering toward the chart's
    retrograde singularity merely to decrease a rate-normalized error.
    """
    longitude = np.arange(orbit_samples) * (2 * np.pi / orbit_samples)
    rates = gauss_matrix(target, longitude, mu)[:, :3, :]
    denominator = np.max(np.sum(rates * rates, axis=-1), axis=0)
    error = elements[:3] - target[:3]
    quotient = float(np.sum(error * error / denominator))
    gradient = np.r_[2 * error / denominator, 0.0, 0.0]

    h, k = elements[3:5]
    ht, kt = target[3:5]
    s2, st2 = 1 + h * h + k * k, 1 + ht * ht + kt * kt
    normal = np.array([2 * k, -2 * h, 1 - h * h - k * k]) / s2
    target_normal = np.array([2 * kt, -2 * ht, 1 - ht * ht - kt * kt]) / st2
    sine = np.linalg.norm(np.cross(normal, target_normal))
    cosine = np.clip(normal @ target_normal, -1.0, 1.0)
    angle = np.arctan2(sine, cosine)
    # The maximum plane-rotation rate per unit normal acceleration is r/|H|,
    # attained at target apoapsis. Its inverse square supplies the plane weight.
    plane_weight = mu / target[0] * (1 - np.hypot(target[1], target[2])) ** 2
    quotient += plane_weight * angle * angle
    normal_gradient = np.array(
        [
            (np.array([0.0, -2.0, -2 * h]) - 2 * h * normal) / s2,
            (np.array([2.0, 0.0, -2 * k]) - 2 * k * normal) / s2,
        ]
    )
    if sine < 1e-12 and cosine < 0:
        # Antipodal planes have equally short rotation paths. Choose the path
        # toward the chart's north pole instead of returning a zero gradient.
        tangent = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
        if np.linalg.norm(tangent) < 1e-12:
            tangent = np.array([1.0, 0.0, 0.0])
        tangent /= np.linalg.norm(tangent)
        gradient[3:5] = -2 * plane_weight * angle * (normal_gradient @ tangent)
    else:
        # theta/sin(theta) tends to one for coincident planes.
        angle_ratio = angle / sine if sine > 1e-12 else 1.0
        tangent = target_normal - cosine * normal
        gradient[3:5] = -2 * plane_weight * angle_ratio * (normal_gradient @ tangent)
    return quotient, gradient


def qlaw_direction(rv, target, mu, orbit_samples):
    """Choose unit RIC thrust minimizing the instantaneous quotient derivative."""
    elements, longitude = equinoctial_elements(rv, mu)
    _, gradient = proximity_quotient_gradient(elements, target, mu, orbit_samples)
    descent = -(gradient @ gauss_matrix(elements, np.asarray(longitude), mu))
    norm = np.linalg.norm(descent)
    return descent / norm if norm > 1e-20 else np.zeros(3)


def resample_seed(history, times):
    """Hermite sample position/velocity together; linearly sample mass/control.

    Linear Cartesian interpolation cuts across orbital arcs and creates large
    collocation defects, even with an accurate propagated seed.
    """
    indices = np.clip(np.searchsorted(history[:, 7], times, side="right") - 1, 0, len(history) - 2)
    left, right = history[indices], history[indices + 1]
    dt = (right[:, 7] - left[:, 7])[:, None]
    u = ((times - left[:, 7])[:, None]) / dt
    r0, v0, r1, v1 = left[:, :3], left[:, 3:6], right[:, :3], right[:, 3:6]
    position = (2 * u**3 - 3 * u**2 + 1) * r0 + (u**3 - 2 * u**2 + u) * dt * v0
    position += (-2 * u**3 + 3 * u**2) * r1 + (u**3 - u**2) * dt * v1
    velocity = (6 * u**2 - 6 * u) * r0 / dt + (3 * u**2 - 4 * u + 1) * v0
    velocity += (-6 * u**2 + 6 * u) * r1 / dt + (3 * u**2 - 2 * u) * v1
    mass = left[:, 6:7] * (1 - u) + right[:, 6:7] * u
    control = left[:, 8:11] * (1 - u) + right[:, 8:11] * u
    return list(np.column_stack([position, velocity, mass, times, control]))


def equinoctial_to_cartesian(elements, longitude, mu):
    """Convert modified equinoctial elements without circular/equatorial singularities."""
    p, f, g, h, k = elements
    s2 = 1 + h * h + k * k
    first = np.array([1 + h * h - k * k, 2 * h * k, -2 * k]) / s2
    second = np.array([2 * h * k, 1 - h * h + k * k, 2 * h]) / s2
    c, sn = np.cos(longitude), np.sin(longitude)
    r = p / (1 + f * c + g * sn) * (c * first + sn * second)
    v = np.sqrt(mu / p) * (-(sn + g) * first + (c + f) * second)
    return np.r_[r, v]


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


def averaged_rates(current, target, mu, thrust, isp, config):
    """Time-average the Gauss rates and mean-longitude rate over one orbit."""
    p, f, g, h, k = current[:5]
    longitude = np.arange(config.orbit_samples) * (2 * np.pi / config.orbit_samples)
    c, sn = np.cos(longitude), np.sin(longitude)
    w = 1 + f * c + g * sn
    matrix = gauss_matrix(current[:5], longitude, mu)
    _, gradient = proximity_quotient_gradient(current[:5], target, mu, config.orbit_samples)
    descent = -np.einsum("i,nij->nj", gradient, matrix)
    norm = np.linalg.norm(descent, axis=1)
    unit = np.divide(
        descent, norm[:, None], out=np.zeros_like(descent), where=norm[:, None] > 1e-20
    )
    acceleration = thrust / current[6] * config.throttle * unit
    rates = np.einsum("nij,nj->ni", matrix, acceleration)
    # Uniform true-longitude samples require Keplerian residence-time weights.
    weights = 1 / w**2
    weights /= np.sum(weights)
    eccentricity = np.hypot(f, g)
    dmean_dlongitude = (1 - eccentricity**2) ** 1.5 / w**2
    if eccentricity < 1e-7:
        dmean_df, dmean_dg = -2 * sn, 2 * c
    else:
        varpi = np.arctan2(g, f)
        nu = longitude - varpi
        eccentric = np.arctan2(np.sqrt(1 - eccentricity**2) * np.sin(nu), eccentricity + np.cos(nu))
        de = (
            -np.sin(eccentric)
            * (2 - eccentricity * np.cos(eccentric) - eccentricity**2)
            / (1 - eccentricity**2)
        )
        dvarpi = 1 - dmean_dlongitude
        dmean_df = de * f / eccentricity - dvarpi * g / eccentricity**2
        dmean_dg = de * g / eccentricity + dvarpi * f / eccentricity**2
    longitude_perturbation = np.sqrt(p / mu) * (h * sn - k * c) / w * acceleration[:, 2]
    mean_rate = np.sqrt(mu / (p / (1 - eccentricity**2)) ** 3)
    mean_rate += weights @ (
        dmean_dlongitude * longitude_perturbation + dmean_df * rates[:, 1] + dmean_dg * rates[:, 2]
    )
    mass_rate = -current[6] / (isp * G0_MPS2) * (weights @ np.linalg.norm(acceleration, axis=1))
    return np.r_[weights @ rates, mean_rate, mass_rate]


def build_low_thrust_qlaw_seed(
    *,
    initial_rv,
    target_rv,
    mu_m3ps2,
    initial_mass_kg,
    dry_mass_kg,
    thrust_N,
    isp_s,
    time_bounds_s,
    npts,
    config,
):
    """Integrate Q-law in equinoctial elements into inertial [R,V,M,t,U] rows.

    Element propagation avoids secular energy drift from integrating Cartesian
    gravity across many revolutions. A partial seed is explicitly reported when
    the upper time bound is reached; propellant exhaustion raises an error.
    """
    mu = float(mu_m3ps2)
    tmin, tmax = map(float, time_bounds_s)
    scalars = [mu, initial_mass_kg, dry_mass_kg, thrust_N, isp_s, tmin, tmax]
    if not np.all(np.isfinite(scalars)) or not (
        mu > 0 and thrust_N > 0 and isp_s > 0 and initial_mass_kg > dry_mass_kg > 0
    ):
        raise ValueError("Q-law requires positive dynamics/propulsion and usable propellant.")
    if not 0 <= tmin < tmax:
        raise ValueError("Q-law time bounds must satisfy max > min >= 0.")
    initial_rv, target_rv = np.asarray(initial_rv, dtype=float), np.asarray(target_rv, dtype=float)
    # Choose a working chart before converting to elements. A large node change
    # can cross the inertial chart's pole; retrograde-equatorial input states
    # are also regular in this frame. Ellipticity is checked after rotation.
    for rv in (initial_rv, target_rv):
        if rv.shape != (6,) or not np.all(np.isfinite(rv)):
            raise ValueError("Q-law requires finite six-component Cartesian states.")
        if np.linalg.norm(rv[:3]) <= 0 or np.linalg.norm(np.cross(rv[:3], rv[3:6])) <= 0:
            raise ValueError("Q-law requires nonzero radius and angular momentum.")
    initial_normal = np.cross(initial_rv[:3], initial_rv[3:6])
    initial_normal /= np.linalg.norm(initial_normal)
    target_normal = np.cross(target_rv[:3], target_rv[3:6])
    target_normal /= np.linalg.norm(target_normal)
    north = initial_normal + target_normal
    if np.linalg.norm(north) < 1e-10:
        north = initial_rv[:3] / np.linalg.norm(initial_rv[:3])
    else:
        north /= np.linalg.norm(north)
    first = initial_rv[:3] - (initial_rv[:3] @ north) * north
    if np.linalg.norm(first) < 1e-10 * np.linalg.norm(initial_rv[:3]):
        axis = np.eye(3)[np.argmin(np.abs(north))]
        first = axis - (axis @ north) * north
    first /= np.linalg.norm(first)
    working_basis = np.column_stack([first, np.cross(north, first), north])
    initial_rv = (initial_rv.reshape(2, 3) @ working_basis).reshape(6)
    target_rv = (target_rv.reshape(2, 3) @ working_basis).reshape(6)
    target, _ = equinoctial_elements(target_rv, mu)
    initial, longitude = equinoctial_elements(initial_rv, mu)
    target_a = target[0] / (1 - target[1] ** 2 - target[2] ** 2)
    target_normal = np.cross(target_rv[:3], target_rv[3:6])
    target_normal /= np.linalg.norm(target_normal)
    target_evec = np.cross(target_rv[3:6], np.cross(target_rv[:3], target_rv[3:6])) / mu
    target_evec -= target_rv[:3] / np.linalg.norm(target_rv[:3])
    state = np.r_[
        initial,
        mean_longitude(initial, longitude) if config.averaged else longitude,
        initial_mass_kg,
    ]
    coast = False
    coast_start_time = None
    evaluations = 0

    def longitude_for(current):
        return true_longitude(current[:5], current[5]) if config.averaged else current[5]

    def errors(current):
        p, f, g = current[:3]
        rv = equinoctial_to_cartesian(current[:5], longitude_for(current), mu)
        r, v = rv[:3], rv[3:6]
        momentum = np.cross(r, v)
        normal = momentum / np.linalg.norm(momentum)
        evec = np.cross(v, momentum) / mu - r / np.linalg.norm(r)
        angle = np.arctan2(np.linalg.norm(np.cross(normal, target_normal)), normal @ target_normal)
        return (
            abs(p / (1 - f * f - g * g) / target_a - 1),
            np.linalg.norm(evec - target_evec),
            float(np.rad2deg(angle)),
        )

    def ric_control(current, coasting=None):
        nonlocal evaluations
        if coasting is True or (coasting is None and coast):
            return np.zeros(3)
        evaluations += 1
        _, gradient = proximity_quotient_gradient(current[:5], target, mu, config.orbit_samples)
        descent = -(gradient @ gauss_matrix(current[:5], np.asarray(longitude_for(current)), mu))
        norm = np.linalg.norm(descent)
        return config.throttle * descent / norm if norm > 1e-20 else np.zeros(3)

    def derivative(current):
        nonlocal evaluations
        if config.averaged:
            if coast:
                p, f, g = current[:3]
                return np.r_[np.zeros(5), np.sqrt(mu / (p / (1 - f * f - g * g)) ** 3), 0]
            evaluations += 1
            return averaged_rates(current, target, mu, thrust_N, isp_s, config)
        p, f, g, h, k, longitude, mass = current
        acceleration = thrust_N / mass * ric_control(current)
        c, sn = np.cos(longitude), np.sin(longitude)
        w = 1 + f * c + g * sn
        element_rate = gauss_matrix(current[:5], np.asarray(longitude), mu) @ acceleration
        longitude_rate = (
            np.sqrt(mu / p**3) * w * w + np.sqrt(p / mu) * (h * sn - k * c) / w * acceleration[2]
        )
        mass_rate = -mass * np.linalg.norm(acceleration) / (isp_s * G0_MPS2)
        return np.r_[element_rate, longitude_rate, mass_rate]

    def row(current, time):
        rv = equinoctial_to_cartesian(current[:5], longitude_for(current), mu)
        radial = rv[:3] / np.linalg.norm(rv[:3])
        normal = np.cross(rv[:3], rv[3:6])
        normal /= np.linalg.norm(normal)
        basis = np.column_stack([radial, np.cross(normal, radial), normal])
        return np.r_[
            (rv.reshape(2, 3) @ working_basis.T).reshape(6),
            current[6],
            time,
            working_basis
            @ basis
            @ ric_control(
                current, coasting=coast_start_time is not None and time >= coast_start_time
            ),
        ]

    time, integration_steps = 0.0, 0
    history = [row(state, time)]
    element_history = [np.r_[state, time]]
    tolerances = (config.semi_major_axis_rtol, config.eccentricity_tol, config.plane_tol_deg)
    while time < tmax:
        if coast and time >= tmin:
            break
        if all(
            error <= tolerance for error, tolerance in zip(errors(state), tolerances, strict=True)
        ):
            if time >= tmin:
                break
            coast = True
            coast_start_time = time
            history[-1][8:11] = 0
        p, f, g = state[:3]
        eccentricity = np.hypot(f, g)
        if p <= 0 or eccentricity >= 1 or not np.all(np.isfinite(state)):
            raise ValueError("Q-law seed left the bound elliptic-orbit domain.")
        period = 2 * np.pi * np.sqrt((p / (1 - eccentricity**2)) ** 3 / mu)
        step = min(
            period / 4
            if config.averaged
            else period * (1 - eccentricity) ** 1.5 / config.steps_per_orbit,
            tmax - time,
        )
        k1 = derivative(state)
        if not coast:
            # Limit terminal overshoot for tight tolerances in either mode.
            component_tolerance = np.array(
                [
                    target[0] * config.semi_major_axis_rtol / 4,
                    config.eccentricity_tol / 4,
                    config.eccentricity_tol / 4,
                    np.deg2rad(config.plane_tol_deg) / 8,
                    np.deg2rad(config.plane_tol_deg) / 8,
                ]
            )
            error = state[:5] - target
            approaching = (error * k1[:5] < 0) & (np.abs(error) > component_tolerance)
            if np.any(approaching):
                step = min(
                    step, float(np.min(0.5 * np.abs(error[approaching] / k1[:5][approaching])))
                )
        if coast:
            step = min(step, tmin - time)
        if (
            not coast
            and state[6] - thrust_N * config.throttle * step / (isp_s * G0_MPS2) <= dry_mass_kg
        ):
            raise ValueError(
                "Q-law seed exhausts usable propellant before reaching the target; "
                "add propellant or reduce the requested orbit change."
            )
        k2 = derivative(state + step * k1 / 2)
        k3 = derivative(state + step * k2 / 2)
        k4 = derivative(state + step * k3)
        state += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        time += step
        history.append(row(state, time))
        element_history.append(np.r_[state, time])
        integration_steps += 1
    if time <= 0:
        raise ValueError("Q-law target already matches the initial orbit; use a coast guess.")
    times = np.linspace(0, time, max(npts, 2))
    if config.averaged:
        dense = np.asarray(element_history)
        interpolated = np.column_stack(
            [np.interp(times, dense[:, 7], dense[:, i]) for i in range(7)]
        )
        # Keep mean longitude unwrapped throughout interpolation.
        rows = [row(current, time) for current, time in zip(interpolated, times, strict=True)]
    else:
        rows = resample_seed(np.asarray(history), times)
    a_error, e_error, plane_error = errors(state)
    return rows, {
        "guess_kind": "low_thrust_qlaw",
        "seed_quotient_normalization": "target_orbit",
        "seed_averaged": config.averaged,
        "seed_tof_s": float(time),
        "seed_throttle": config.throttle,
        "seed_final_mass_kg": float(state[6]),
        "seed_integration_steps": integration_steps,
        "seed_control_evaluations": evaluations,
        "seed_estimated_revolutions": float(
            (state[5] - (mean_longitude(initial, longitude) if config.averaged else longitude))
            / (2 * np.pi)
        ),
        "seed_target_reached": bool(
            all(
                error <= tolerance
                for error, tolerance in zip(
                    (a_error, e_error, plane_error), tolerances, strict=True
                )
            )
        ),
        "seed_semi_major_axis_relative_error": float(a_error),
        "seed_eccentricity_vector_error": float(e_error),
        "seed_plane_error_deg": plane_error,
    }
