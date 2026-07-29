"""Coupled absolute-state propagation for perturbed relative motion.

The public result remains chief-centered RIC, while the numerical integration
advances chief and deputy absolute states under the same force model.  Taking
the difference only after propagation naturally retains differential J2 and
third-body accelerations without embedding frame-rotation approximations in
the force model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..bodies import EARTH, MOON, SUN, CelestialBody
from ..data.ephemeris import (
    DEFAULT_EPHEMERIS_BSP,
    epoch_to_et,
    sample_sun_moon_positions_eci_tod,
)
from ..dynamics import j2_acceleration_components, third_body_acceleration_components
from ..specs import BoundaryState
from .transforms import (
    inertial_to_relative_state,
    relative_to_inertial_state,
    ric_basis,
)

StateHistory = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RelativePropagationResult:
    """Coupled chief/deputy propagation histories.

    Attributes:
        times_s: Elapsed seconds from the supplied initial states.
        chief_states_eci: Absolute chief states with shape ``(N, 6)``.
        deputy_states_eci: Absolute deputy states with shape ``(N, 6)``.
        relative_states_ric: Deputy-minus-chief RIC states with shape ``(N, 6)``.
    """

    times_s: NDArray[np.float64]
    chief_states_eci: StateHistory
    deputy_states_eci: StateHistory
    relative_states_ric: StateHistory

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float).reshape(-1)
        chief = np.asarray(self.chief_states_eci, dtype=float)
        deputy = np.asarray(self.deputy_states_eci, dtype=float)
        relative = np.asarray(self.relative_states_ric, dtype=float)
        expected = (times.size, 6)
        for name, value in (
            ("chief_states_eci", chief),
            ("deputy_states_eci", deputy),
            ("relative_states_ric", relative),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "chief_states_eci", chief)
        object.__setattr__(self, "deputy_states_eci", deputy)
        object.__setattr__(self, "relative_states_ric", relative)

    @property
    def chief_trajectory_eci(self) -> StateHistory:
        """Return ``[chief ECI state, time]`` rows."""
        return np.column_stack([self.chief_states_eci, self.times_s])

    @property
    def deputy_trajectory_eci(self) -> StateHistory:
        """Return ``[deputy ECI state, time]`` rows."""
        return np.column_stack([self.deputy_states_eci, self.times_s])

    @property
    def relative_trajectory_ric(self) -> StateHistory:
        """Return ``[relative RIC state, time]`` rows ready for plotting."""
        return np.column_stack([self.relative_states_ric, self.times_s])


def propagate_relative_numerical(
    chief_initial_eci: BoundaryState,
    relative_initial_ric: BoundaryState | None,
    times_s: ArrayLike,
    *,
    deputy_initial_eci: BoundaryState | None = None,
    central_body: CelestialBody = EARTH,
    perturbations: Any | None = None,
    initial_epoch: str | datetime | float | int | None = None,
    max_step_s: float = 10.0,
    ephemeris_step_s: float = 600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
) -> RelativePropagationResult:
    """Propagate chief and deputy together and report their RIC separation.

    The integrator is fixed-step fourth-order Runge-Kutta with exact output at
    each requested time.  Both spacecraft receive central gravity plus the
    requested J2, lunar, and solar accelerations.  This coupled absolute-state
    formulation is deliberately simple and physically consistent for a first
    perturbed relative-motion capability.

    Args:
        chief_initial_eci: Chief absolute Cartesian state at elapsed time zero.
        relative_initial_ric: Deputy state relative to the chief in RIC. Pass
            ``None`` when supplying ``deputy_initial_eci`` directly.
        times_s: Strictly monotonic output times with zero at either endpoint.
            Forward histories normally begin at zero; backward histories may
            be requested as either ``[0, ..., negative]`` or
            ``[negative, ..., 0]``.
        deputy_initial_eci: Optional absolute deputy state at time zero. This
            avoids a frame-velocity round trip when osculating elements define
            the deputy under a perturbed force model. Exactly one deputy
            initial-state representation must be supplied.
        central_body: Central-body constants used by gravity and J2.
        perturbations: An :class:`octavian.Perturbations` configuration.  J2,
            Moon, and Sun are supported; drag and SRP raise explicitly.
        initial_epoch: UTC or SPICE ET corresponding to time zero.  Required
            when lunar or solar gravity is enabled.
        max_step_s: Maximum internal RK4 step in seconds.
        ephemeris_step_s: Sun/Moon interpolation sample spacing in seconds.
        bsp_path: SPICE BSP containing Earth-centered Sun/Moon states.

    Returns:
        Absolute chief/deputy histories and the equivalent RIC history.
    """
    requested_times = np.asarray(times_s, dtype=float).reshape(-1)
    if requested_times.size < 1 or not np.all(np.isfinite(requested_times)):
        raise ValueError("times_s must contain at least one finite value")
    if np.isclose(requested_times[0], 0.0, atol=1.0e-12):
        integration_times = requested_times
        reverse_output = False
    elif np.isclose(requested_times[-1], 0.0, atol=1.0e-12):
        integration_times = requested_times[::-1]
        reverse_output = True
    else:
        raise ValueError("times_s must have 0.0 at the first or last output")
    if integration_times.size > 1:
        differences = np.diff(integration_times)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError("times_s must be strictly monotonic")
    maximum_step = float(max_step_s)
    ephemeris_step = float(ephemeris_step_s)
    if not np.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("max_step_s must be finite and positive")
    if not np.isfinite(ephemeris_step) or ephemeris_step <= 0.0:
        raise ValueError("ephemeris_step_s must be finite and positive")
    if (relative_initial_ric is None) == (deputy_initial_eci is None):
        raise ValueError(
            "Supply exactly one of relative_initial_ric or deputy_initial_eci"
        )

    flags = _normalize_perturbations(perturbations)
    requested_bodies = flags["third_bodies"]
    body_positions = _build_body_position_interpolator(
        requested_bodies=requested_bodies,
        initial_epoch=initial_epoch,
        start_time_s=float(np.min(integration_times)),
        end_time_s=float(np.max(integration_times)),
        step_s=ephemeris_step,
        bsp_path=bsp_path,
    )

    initial_chief_acceleration = _absolute_acceleration(
        chief_initial_eci.r_m,
        central_body=central_body,
        include_j2=flags["j2"],
        third_bodies=body_positions(0.0),
    )
    if deputy_initial_eci is None:
        if relative_initial_ric is None:  # guarded by the exclusive-input check
            raise RuntimeError("Relative initial state validation failed")
        deputy_initial_eci = relative_to_inertial_state(
            chief_initial_eci,
            relative_initial_ric,
            chief_acceleration_mps2=initial_chief_acceleration,
        )
    combined_state = np.hstack(
        [
            chief_initial_eci.r_m,
            chief_initial_eci.v_mps,
            deputy_initial_eci.r_m,
            deputy_initial_eci.v_mps,
        ]
    ).astype(float)
    absolute_history = np.empty((integration_times.size, 12), dtype=float)
    absolute_history[0] = combined_state

    def derivative(time_s: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the stacked chief/deputy absolute derivative."""
        chief_position = state[0:3]
        chief_velocity = state[3:6]
        deputy_position = state[6:9]
        deputy_velocity = state[9:12]
        sampled_bodies = body_positions(time_s)
        chief_acceleration = _absolute_acceleration(
            chief_position,
            central_body=central_body,
            include_j2=flags["j2"],
            third_bodies=sampled_bodies,
        )
        deputy_acceleration = _absolute_acceleration(
            deputy_position,
            central_body=central_body,
            include_j2=flags["j2"],
            third_bodies=sampled_bodies,
        )
        return np.hstack(
            [
                chief_velocity,
                chief_acceleration,
                deputy_velocity,
                deputy_acceleration,
            ]
        )

    current_time = 0.0
    for output_index, output_time in enumerate(integration_times[1:], start=1):
        interval = float(output_time - current_time)
        substeps = max(1, int(np.ceil(abs(interval) / maximum_step)))
        step = interval / substeps
        for _ in range(substeps):
            k1 = derivative(current_time, combined_state)
            k2 = derivative(current_time + 0.5 * step, combined_state + 0.5 * step * k1)
            k3 = derivative(current_time + 0.5 * step, combined_state + 0.5 * step * k2)
            k4 = derivative(current_time + step, combined_state + step * k3)
            combined_state = combined_state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current_time += step
        current_time = float(output_time)
        if not np.all(np.isfinite(combined_state)):
            raise RuntimeError(f"Relative propagation became non-finite at t={current_time:.6f} s")
        absolute_history[output_index] = combined_state

    chief_states = absolute_history[:, 0:6]
    deputy_states = absolute_history[:, 6:12]
    relative_states = np.empty_like(chief_states)
    for index, time_s in enumerate(integration_times):
        chief = BoundaryState(chief_states[index, 0:3], chief_states[index, 3:6])
        deputy = BoundaryState(
            deputy_states[index, 0:3],
            deputy_states[index, 3:6],
        )
        chief_acceleration = _absolute_acceleration(
            chief.r_m,
            central_body=central_body,
            include_j2=flags["j2"],
            third_bodies=body_positions(float(time_s)),
        )
        relative = inertial_to_relative_state(
            chief,
            deputy,
            chief_acceleration_mps2=chief_acceleration,
        )
        relative_states[index] = np.hstack([relative.r_m, relative.v_mps])
    if reverse_output:
        chief_states = chief_states[::-1].copy()
        deputy_states = deputy_states[::-1].copy()
        relative_states = relative_states[::-1].copy()
    return RelativePropagationResult(
        times_s=requested_times,
        chief_states_eci=chief_states,
        deputy_states_eci=deputy_states,
        relative_states_ric=relative_states,
    )


def nonlinear_relative_ric_derivative(
    relative_state_ric: ArrayLike,
    *,
    mu_m3ps2: float,
    chief_orbit_radius_m: float,
) -> NDArray[np.float64]:
    """Evaluate exact circular-chief RIC dynamics before linearization.

    Args:
        relative_state_ric: ``[R, I, C, Rdot, Idot, Cdot]`` in SI units.
        mu_m3ps2: Central-body gravitational parameter.
        chief_orbit_radius_m: Circular chief radius in meters.

    Returns:
        Six state derivatives in SI units.
    """
    state = np.asarray(relative_state_ric, dtype=float).reshape(6)
    mu = float(mu_m3ps2)
    chief_radius = float(chief_orbit_radius_m)
    if not np.all(np.isfinite(state)):
        raise ValueError("relative_state_ric must contain finite values")
    if mu <= 0.0 or chief_radius <= 0.0:
        raise ValueError("mu_m3ps2 and chief_orbit_radius_m must be positive")
    x, y, z, xdot, ydot, zdot = state
    deputy_radius = float(np.sqrt((chief_radius + x) ** 2 + y**2 + z**2))
    if deputy_radius <= 0.0:
        raise ValueError("Relative state places the deputy at the central body")
    mean_motion = float(np.sqrt(mu / chief_radius**3))
    gravity_rate = mu / deputy_radius**3
    return np.asarray(
        [
            xdot,
            ydot,
            zdot,
            2.0 * mean_motion * ydot + (chief_radius + x) * (mean_motion**2 - gravity_rate),
            -2.0 * mean_motion * xdot + y * (mean_motion**2 - gravity_rate),
            -gravity_rate * z,
        ],
        dtype=float,
    )


def coupled_relative_ric_derivative(
    stacked_state: ArrayLike,
    *,
    mu_m3ps2: float,
) -> NDArray[np.float64]:
    """Evaluate exact two-body chief-ECI/deputy-RIC stacked dynamics.

    Args:
        stacked_state: ``[chief_r, chief_v, rho_RIC, rho_dot_RIC]`` in SI.
        mu_m3ps2: Central-body gravitational parameter.

    Returns:
        Twelve derivatives in the same state ordering.
    """
    state = np.asarray(stacked_state, dtype=float).reshape(12)
    mu = float(mu_m3ps2)
    if not np.all(np.isfinite(state)):
        raise ValueError("stacked_state must contain finite values")
    if mu <= 0.0:
        raise ValueError("mu_m3ps2 must be positive")
    chief = BoundaryState(state[0:3], state[3:6])
    relative = BoundaryState(state[6:9], state[9:12])
    deputy = relative_to_inertial_state(chief, relative)
    chief_radius = float(np.linalg.norm(chief.r_m))
    deputy_radius = float(np.linalg.norm(deputy.r_m))
    if chief_radius <= 0.0 or deputy_radius <= 0.0:
        raise ValueError("Chief and deputy positions must have non-zero norm")

    chief_acceleration = -mu * chief.r_m / chief_radius**3
    deputy_acceleration = -mu * deputy.r_m / deputy_radius**3
    basis = ric_basis(chief.r_m, chief.v_mps)
    gravity_difference_ric = basis @ (
        deputy_acceleration - chief_acceleration
    )
    angular_momentum = float(np.linalg.norm(np.cross(chief.r_m, chief.v_mps)))
    frame_rate = angular_momentum / chief_radius**2
    frame_rate_derivative = (
        -2.0
        * frame_rate
        * float(np.dot(chief.r_m, chief.v_mps))
        / chief_radius**2
    )
    x, y, _ = relative.r_m
    xdot, ydot, _ = relative.v_mps
    rotating_terms = np.asarray(
        [
            2.0 * frame_rate * ydot
            + frame_rate_derivative * y
            + frame_rate**2 * x,
            -2.0 * frame_rate * xdot
            - frame_rate_derivative * x
            + frame_rate**2 * y,
            0.0,
        ],
        dtype=float,
    )
    return np.hstack(
        [
            chief.v_mps,
            chief_acceleration,
            relative.v_mps,
            gravity_difference_ric + rotating_terms,
        ]
    )


def propagate_nonlinear_relative_ric(
    initial_state_ric: ArrayLike,
    times_s: ArrayLike,
    *,
    mu_m3ps2: float,
    chief_orbit_radius_m: float,
    max_step_s: float = 10.0,
) -> NDArray[np.float64]:
    """Propagate the exact autonomous circular-chief RIC equations with RK4.

    Args:
        initial_state_ric: Initial six-state RIC vector in SI units.
        times_s: Strictly increasing elapsed times beginning at zero.
        mu_m3ps2: Central-body gravitational parameter.
        chief_orbit_radius_m: Circular chief radius in meters.
        max_step_s: Maximum internal Runge-Kutta step.

    Returns:
        An ``(N, 7)`` history containing RIC state and elapsed time.
    """
    output_times = np.asarray(times_s, dtype=float).reshape(-1)
    state = np.asarray(initial_state_ric, dtype=float).reshape(6)
    maximum_step = float(max_step_s)
    if output_times.size == 0 or not np.all(np.isfinite(output_times)):
        raise ValueError("times_s must contain at least one finite value")
    if not np.isclose(output_times[0], 0.0, atol=1.0e-12):
        raise ValueError("times_s must begin at 0.0")
    if output_times.size > 1 and np.any(np.diff(output_times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")
    if maximum_step <= 0.0 or not np.isfinite(maximum_step):
        raise ValueError("max_step_s must be finite and positive")
    history = np.empty((output_times.size, 7), dtype=float)
    history[0] = np.hstack([state, output_times[0]])

    def derivative(value: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the autonomous RIC derivative for one RK4 stage."""
        return nonlinear_relative_ric_derivative(
            value,
            mu_m3ps2=mu_m3ps2,
            chief_orbit_radius_m=chief_orbit_radius_m,
        )

    current_time = 0.0
    for output_index, output_time in enumerate(output_times[1:], start=1):
        interval = float(output_time - current_time)
        substeps = max(1, int(np.ceil(interval / maximum_step)))
        step = interval / substeps
        for _ in range(substeps):
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * step * k1)
            k3 = derivative(state + 0.5 * step * k2)
            k4 = derivative(state + step * k3)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        current_time = float(output_time)
        history[output_index] = np.hstack([state, current_time])
    return history


def _normalize_perturbations(perturbations: Any | None) -> dict[str, Any]:
    if perturbations is None:
        return {"j2": False, "third_bodies": ()}
    unsupported = [name for name in ("srp", "drag") if bool(getattr(perturbations, name, False))]
    if unsupported:
        raise NotImplementedError(
            "Perturbed relative propagation currently supports J2, Moon, and Sun; "
            f"unsupported flags: {', '.join(unsupported)}"
        )
    if hasattr(perturbations, "active_third_bodies"):
        bodies = tuple(perturbations.active_third_bodies())
    else:
        bodies = tuple(
            str(name).strip().lower() for name in getattr(perturbations, "third_bodies", ())
        )
        for enabled_name in ("moon", "sun"):
            if bool(getattr(perturbations, enabled_name, False)) and enabled_name not in bodies:
                bodies += (enabled_name,)
    unsupported_bodies = [name for name in bodies if name not in {"moon", "sun"}]
    if unsupported_bodies:
        raise NotImplementedError(
            "Perturbed relative propagation supports Moon and Sun third-body gravity; "
            f"unsupported bodies: {', '.join(unsupported_bodies)}"
        )
    return {"j2": bool(getattr(perturbations, "j2", False)), "third_bodies": bodies}


def _absolute_acceleration(
    position_m: NDArray[np.float64],
    *,
    central_body: CelestialBody,
    include_j2: bool,
    third_bodies: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    radius = float(np.linalg.norm(position_m))
    if radius <= 0.0:
        raise ValueError("Absolute spacecraft position must have non-zero norm")
    acceleration = -float(central_body.mu_m3ps2) * position_m / radius**3
    if include_j2:
        acceleration += np.asarray(
            j2_acceleration_components(
                position_m,
                mu_m3ps2=central_body.mu_m3ps2,
                radius_m=central_body.mean_radius_m,
                j2=central_body.j2_coefficient,
            ),
            dtype=float,
        )
    third_body_catalog = {"moon": MOON, "sun": SUN}
    for name, body_position_m in third_bodies.items():
        acceleration += np.asarray(
            third_body_acceleration_components(
                position_m,
                body_position_m,
                mu_m3ps2=third_body_catalog[name].mu_m3ps2,
            ),
            dtype=float,
        )
    return acceleration


def _build_body_position_interpolator(
    *,
    requested_bodies: tuple[str, ...],
    initial_epoch: str | datetime | float | int | None,
    start_time_s: float,
    end_time_s: float,
    step_s: float,
    bsp_path: str | Path,
):
    if not requested_bodies:
        return lambda _time_s: {}
    if initial_epoch is None:
        raise ValueError("initial_epoch is required for Moon or Sun relative perturbations")
    duration_s = float(end_time_s - start_time_s)
    if duration_s <= 0.0:
        raise ValueError("Moon or Sun perturbations require a nonzero propagation span")
    table_times, table_positions = sample_sun_moon_positions_eci_tod(
        initial_epoch=epoch_to_et(initial_epoch) + float(start_time_s),
        duration_s=duration_s,
        step_s=step_s,
        bsp_path=bsp_path,
    )
    table_times = table_times + float(start_time_s)

    def interpolate(time_s: float) -> dict[str, NDArray[np.float64]]:
        """Interpolate requested third-body positions at elapsed time."""
        return {
            name: np.asarray(
                [
                    np.interp(time_s, table_times, table_positions[name][:, component])
                    for component in range(3)
                ],
                dtype=float,
            )
            for name in requested_bodies
        }

    return interpolate
