"""Initial guesses for finite- and low-thrust phases."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...astro.types import as_vec3
from ...guesses import LowThrustSpiralGuess

G0_MPS2 = 9.80665


def circular_spiral_delta_v_mps(
    initial_radius_m: float,
    target_radius_m: float,
    mu_m3ps2: float,
) -> float:
    """Estimate tangential delta-v for a near-circular orbit spiral."""
    r0 = float(initial_radius_m)
    rf = float(target_radius_m)
    mu = float(mu_m3ps2)
    if r0 <= 0.0 or rf <= 0.0 or mu <= 0.0:
        raise ValueError("Spiral radii and gravitational parameter must be positive.")
    return abs(float(np.sqrt(mu / r0) - np.sqrt(mu / rf)))


def constant_throttle_burn_time_s(
    delta_v_mps: float,
    *,
    initial_mass_kg: float,
    thrust_N: float,
    isp_s: float,
    throttle: float,
) -> float:
    """Estimate burn time from the rocket equation at constant throttle."""
    delta_v = max(float(delta_v_mps), 0.0)
    mass0 = float(initial_mass_kg)
    thrust = float(thrust_N)
    isp = float(isp_s)
    throttle_value = float(throttle)
    if mass0 <= 0.0 or thrust <= 0.0 or isp <= 0.0:
        raise ValueError("Initial mass, thrust, and specific impulse must be positive.")
    if not (0.0 < throttle_value <= 1.0):
        raise ValueError("Throttle must be in (0, 1].")
    final_mass = mass0 * float(np.exp(-delta_v / (isp * G0_MPS2)))
    propellant_kg = mass0 - final_mass
    mass_flow_kgps = thrust * throttle_value / (isp * G0_MPS2)
    return propellant_kg / mass_flow_kgps


def build_low_thrust_spiral_seed(
    *,
    initial_position_m: Sequence[float],
    initial_velocity_mps: Sequence[float],
    target_radius_m: float,
    mu_m3ps2: float,
    initial_mass_kg: float,
    dry_mass_kg: float,
    thrust_N: float,
    isp_s: float,
    time_bounds_s: tuple[float, float],
    npts: int,
    config: LowThrustSpiralGuess,
) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
    """Integrate a constant-throttle tangential spiral into ASSET guess rows.

    Rows use the powered layout ``[R(3), V(3), M, t, U(3)]``. This is a seed,
    not an imposed steering law; all control samples and times remain decision
    variables in the compiled optimal-control problem.
    """
    r0 = as_vec3(initial_position_m)
    v0 = as_vec3(initial_velocity_mps)
    radius0 = float(np.linalg.norm(r0))
    target_radius = float(target_radius_m)
    mu = float(mu_m3ps2)
    mass0 = float(initial_mass_kg)
    dry_mass = float(dry_mass_kg)
    thrust = float(thrust_N)
    isp = float(isp_s)
    throttle = float(config.throttle)
    if radius0 <= 0.0 or float(np.linalg.norm(v0)) <= 0.0:
        raise ValueError("Low-thrust spiral seeding requires nonzero position and velocity.")
    if abs(target_radius - radius0) <= max(radius0, 1.0) * 1e-10:
        raise ValueError("Low-thrust spiral seeding requires a distinct target radius.")
    if mass0 <= dry_mass:
        raise ValueError("Low-thrust spiral seeding requires positive usable propellant mass.")

    direction = str(config.direction).strip().lower()
    if direction == "auto":
        direction = "prograde" if target_radius > radius0 else "retrograde"
    direction_sign = 1.0 if direction == "prograde" else -1.0

    delta_v_estimate = circular_spiral_delta_v_mps(radius0, target_radius, mu)
    burn_time_estimate = constant_throttle_burn_time_s(
        delta_v_estimate,
        initial_mass_kg=mass0,
        thrust_N=thrust,
        isp_s=isp,
        throttle=throttle,
    )
    tmin, tmax = map(float, time_bounds_s)
    if not (tmax > tmin >= 0.0):
        raise ValueError("Low-thrust spiral time bounds must satisfy max > min >= 0.")
    tf = float(np.clip(burn_time_estimate * float(config.time_scale), tmin, tmax))
    mass_flow_kgps = thrust * throttle / (isp * G0_MPS2)
    final_mass_estimate = mass0 - mass_flow_kgps * tf
    if final_mass_estimate < dry_mass:
        raise ValueError(
            "The low-thrust seed would consume more propellant than the spacecraft carries; "
            "shorten the time bounds, reduce seed throttle, or add propellant."
        )

    output_times = np.linspace(0.0, tf, max(int(npts), 2))
    initial_period_s = float(2.0 * np.pi * np.sqrt(radius0**3 / mu))
    maximum_step_s = initial_period_s / int(config.steps_per_orbit)
    state = np.hstack([r0, v0, mass0]).astype(float)
    rows: list[np.ndarray] = []
    integration_steps = 0

    def control_for(current_state: np.ndarray) -> np.ndarray:
        velocity = current_state[3:6]
        speed = float(np.linalg.norm(velocity))
        if speed <= 0.0:
            raise ValueError("Low-thrust spiral integration encountered zero velocity.")
        return direction_sign * throttle * velocity / speed

    def derivative(current_state: np.ndarray) -> np.ndarray:
        position = current_state[0:3]
        mass = max(float(current_state[6]), dry_mass)
        radius = float(np.linalg.norm(position))
        gravity = -mu * position / radius**3
        control = control_for(current_state)
        thrust_acceleration = thrust * control / mass
        mass_rate = -thrust / (isp * G0_MPS2) * float(np.linalg.norm(control))
        return np.hstack([current_state[3:6], gravity + thrust_acceleration, mass_rate])

    previous_time = 0.0
    rows.append(np.hstack([state[0:6], state[6], 0.0, control_for(state)]))
    for output_time in output_times[1:]:
        interval = float(output_time - previous_time)
        substeps = max(1, int(np.ceil(interval / maximum_step_s)))
        step = interval / substeps
        for _ in range(substeps):
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * step * k1)
            k3 = derivative(state + 0.5 * step * k2)
            k4 = derivative(state + step * k3)
            state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            integration_steps += 1
        previous_time = float(output_time)
        rows.append(
            np.hstack(
                [state[0:6], state[6], float(output_time), control_for(state)]
            )
        )

    return rows, {
        "guess_kind": "low_thrust_tangential_spiral",
        "seed_throttle": throttle,
        "seed_direction": direction,
        "seed_delta_v_estimate_mps": delta_v_estimate,
        "seed_burn_time_estimate_s": burn_time_estimate,
        "seed_tof_s": tf,
        "seed_final_mass_kg": float(state[6]),
        "seed_final_radius_m": float(np.linalg.norm(state[0:3])),
        "seed_integration_steps": integration_steps,
    }
