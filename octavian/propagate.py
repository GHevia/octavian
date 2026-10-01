"""One discoverable namespace for Octavian's analysis propagators.

Use this module as ``from octavian import propagate``.  Existing specialized
functions remain available in :mod:`octavian.relative`, :mod:`octavian.astro`,
and :mod:`octavian.cislunar`; this namespace provides consistent history
outputs and a shorter path for common analysis workflows.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._asset import ast
from .astro import propagate_cartesian_rv
from .bodies import EARTH, CelestialBody
from .data.ephemeris import DEFAULT_EPHEMERIS_BSP
from .models import Perturbations
from .relative import (
    ClassicalRelativeOrbitalElements,
    RelativeElementPropagationResult,
    RelativeOrbitalElements,
    RelativePropagationResult,
    propagate_relative_element_history,
    propagate_two_body_state,
)
from .relative import propagate_cwh as _propagate_cwh
from .relative import (
    propagate_nonlinear_relative_ric as _propagate_nonlinear_relative_ric,
)
from .relative import propagate_relative_numerical as _propagate_relative_numerical
from .relative.propagation import (
    _absolute_acceleration,
    _build_body_position_interpolator,
    _normalize_perturbations,
    _resolved_atmosphere,
    _validate_cannonball_spacecraft,
)
from .spacecraft import Spacecraft
from .specs import BoundaryState

if TYPE_CHECKING:
    from .cislunar import CR3BPSystem

StateHistory = NDArray[np.float64]


def _finite_times(times_s: ArrayLike) -> NDArray[np.float64]:
    """Return a non-empty one-dimensional array of finite elapsed times."""
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain at least one finite value")
    return times


def inertial(
    initial_state: BoundaryState,
    times_s: ArrayLike,
    *,
    central_body: CelestialBody = EARTH,
    perturbations: Perturbations | None = None,
    initial_epoch: str | datetime | float | int | None = None,
    spacecraft: Spacecraft | None = None,
    max_step_s: float = 10.0,
    ephemeris_step_s: float = 600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
) -> StateHistory:
    """Numerically propagate an unpowered inertial Cartesian state.

    Uses the same force model as :func:`relative` and fixed-step RK4, with
    shortened steps to hit each output time exactly. Circular and equatorial
    states are integrated directly without singular orbital elements.

    Args:
        initial_state: Central-body inertial position/velocity at time zero.
        times_s: Strictly monotonic elapsed seconds, with zero at either end.
            Forward and backward propagation are supported.
        central_body: Gravity/J2 defaults; defaults to Earth. A harmonic field
            with published GM takes precedence for central and harmonic gravity.
        perturbations: Optional spherical harmonics or J2, Moon/Sun gravity, drag, and SRP.
        initial_epoch: UTC or SPICE ET at time zero, required for third-body
            gravity or SRP. Ephemeris positions use the Earth-centered TOD frame.
        spacecraft: Constant mass and cannonball properties, required for
            drag or SRP. This analysis propagator does not apply thrust.
        max_step_s: Maximum internal RK4 step in seconds.
        ephemeris_step_s: Sun/Moon interpolation spacing in seconds.
        bsp_path: BSP containing Earth-centered Sun/Moon states.

    Returns:
        ``(N, 7)`` rows ``[r, v, elapsed_time]`` in SI units, in requested order.
    """
    times = _finite_times(times_s)
    if times[0] == 0.0:
        integration_times = times
    elif times[-1] == 0.0:
        integration_times = times[::-1]
    else:
        raise ValueError("times_s must have 0.0 at the first or last output")
    differences = np.diff(integration_times)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError("times_s must be strictly monotonic")
    if not np.isfinite(max_step_s) or max_step_s <= 0.0:
        raise ValueError("max_step_s must be finite and positive")
    if not np.isfinite(ephemeris_step_s) or ephemeris_step_s <= 0.0:
        raise ValueError("ephemeris_step_s must be finite and positive")
    current_state = np.hstack([initial_state.r_m, initial_state.v_mps]).astype(float)
    if not np.all(np.isfinite(current_state)) or np.linalg.norm(current_state[:3]) == 0.0:
        raise ValueError("initial_state must be finite with nonzero position")
    flags = _normalize_perturbations(perturbations)
    if (flags["third_bodies"] or flags["srp"]) and central_body.name.lower() != "earth":
        raise ValueError("The bundled Sun/Moon BSP requires an Earth central body")
    _validate_cannonball_spacecraft(
        spacecraft,
        flags=flags,
        role="inertial",
        required=bool(flags["drag"] or flags["srp"]),
    )
    atmosphere = _resolved_atmosphere(flags=flags, central_body=central_body)
    requested_bodies = list(flags["third_bodies"])
    if flags["srp"] and "sun" not in requested_bodies:
        requested_bodies.append("sun")
    body_positions = _build_body_position_interpolator(
        requested_bodies=tuple(requested_bodies),
        initial_epoch=initial_epoch,
        start_time_s=float(times.min()),
        end_time_s=float(times.max()),
        step_s=ephemeris_step_s,
        bsp_path=bsp_path,
    )

    def derivative(time_s: float, state: np.ndarray) -> np.ndarray:
        acceleration = _absolute_acceleration(
            state[:3],
            state[3:6],
            central_body=central_body,
            flags=flags,
            sampled_bodies=body_positions(time_s),
            time_s=time_s,
            spacecraft=spacecraft,
            atmosphere=atmosphere,
        )
        return np.hstack([state[3:6], acceleration])

    history = np.empty((times.size, 7), dtype=float)
    history[0] = np.hstack([current_state, 0.0])
    current_time = 0.0
    for index, output_time in enumerate(integration_times[1:], start=1):
        interval = float(output_time - current_time)
        substeps = max(1, int(np.ceil(abs(interval) / max_step_s)))
        step = interval / substeps
        for _ in range(substeps):
            k1 = derivative(current_time, current_state)
            k2 = derivative(current_time + step / 2, current_state + step * k1 / 2)
            k3 = derivative(current_time + step / 2, current_state + step * k2 / 2)
            k4 = derivative(current_time + step, current_state + step * k3)
            current_state += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            current_time += step
        current_time = float(output_time)
        if not np.all(np.isfinite(current_state)):
            raise RuntimeError(f"Inertial propagation became non-finite at t={current_time:.6f} s")
        history[index] = np.hstack([current_state, current_time])
    return history if times[0] == 0.0 else history[::-1].copy()


def two_body(
    initial_state: BoundaryState,
    times_s: ArrayLike,
    *,
    mu_m3ps2: float,
) -> StateHistory:
    """Propagate an elliptic inertial Cartesian state at elapsed times.

    Args:
        initial_state: Cartesian state at elapsed time zero.
        times_s: Finite elapsed output times. Analytical evaluation permits
            either order and negative times.
        mu_m3ps2: Central-body gravitational parameter.

    Returns:
        ``(N, 7)`` rows of ``[r, v, elapsed_time]`` in SI units.
    """
    times = _finite_times(times_s)
    history = np.empty((times.size, 7), dtype=float)
    initial = np.hstack([initial_state.r_m, initial_state.v_mps])
    for index, time_s in enumerate(times):
        if ast is not None:
            # Cartesian Kepler propagation avoids circular/equatorial element singularities.
            propagated = propagate_cartesian_rv(initial, float(time_s), float(mu_m3ps2))
            history[index] = np.hstack([propagated, float(time_s)])
        else:
            propagated_state = propagate_two_body_state(
                initial_state, float(time_s), float(mu_m3ps2)
            )
            history[index] = np.hstack(
                [propagated_state.r_m, propagated_state.v_mps, float(time_s)]
            )

    return history


def cwh(
    initial_state_ric: ArrayLike,
    times_s: ArrayLike,
    *,
    mean_motion_radps: float,
) -> StateHistory:
    """Propagate a relative RIC state with the analytic CWH solution.

    Args:
        initial_state_ric: Six-component RIC state at elapsed time zero.
        times_s: Finite elapsed output times.
        mean_motion_radps: Circular chief mean motion.

    Returns:
        ``(N, 7)`` rows of ``[RIC state, elapsed_time]``.
    """
    initial = np.asarray(initial_state_ric, dtype=float).reshape(6)
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial_state_ric must contain finite values")
    times = _finite_times(times_s)
    history = np.empty((times.size, 7), dtype=float)
    for index, time_s in enumerate(times):
        history[index] = np.hstack(
            [
                _propagate_cwh(
                    initial,
                    float(time_s),
                    float(mean_motion_radps),
                ),
                float(time_s),
            ]
        )
    return history


def nonlinear_ric(
    initial_state_ric: ArrayLike,
    times_s: ArrayLike,
    *,
    mu_m3ps2: float,
    chief_orbit_radius_m: float,
    max_step_s: float = 10.0,
) -> StateHistory:
    """Propagate the exact circular-chief RIC equations before linearization.

    Args:
        initial_state_ric: Six-component RIC state at elapsed time zero.
        times_s: Strictly increasing elapsed times beginning at zero.
        mu_m3ps2: Central-body gravitational parameter.
        chief_orbit_radius_m: Circular chief radius.
        max_step_s: Maximum internal RK4 step.

    Returns:
        ``(N, 7)`` rows of ``[RIC state, elapsed_time]``.
    """
    return _propagate_nonlinear_relative_ric(
        initial_state_ric,
        times_s,
        mu_m3ps2=mu_m3ps2,
        chief_orbit_radius_m=chief_orbit_radius_m,
        max_step_s=max_step_s,
    )


def relative(
    chief_initial_eci: BoundaryState,
    relative_initial_ric: BoundaryState | None,
    times_s: ArrayLike,
    *,
    deputy_initial_eci: BoundaryState | None = None,
    central_body: CelestialBody = EARTH,
    perturbations: Perturbations | None = None,
    initial_epoch: str | datetime | float | int | None = None,
    max_step_s: float = 10.0,
    ephemeris_step_s: float = 600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
    chief_spacecraft: Spacecraft | None = None,
    deputy_spacecraft: Spacecraft | None = None,
) -> RelativePropagationResult:
    """Propagate exact coupled chief/deputy states and report RIC motion.

    Args:
        chief_initial_eci: Chief absolute state at elapsed time zero.
        relative_initial_ric: Deputy state in chief RIC axes. Pass ``None``
            when supplying ``deputy_initial_eci``.
        times_s: Strictly monotonic times with zero at one endpoint.
        deputy_initial_eci: Optional absolute deputy state at time zero.
        central_body: Central-body gravity and radius constants.
        perturbations: Optional J2, third-body, drag, and SRP model.
        initial_epoch: Required for Moon, Sun, or SRP.
        max_step_s: Maximum internal RK4 step.
        ephemeris_step_s: Sun/Moon interpolation spacing.
        bsp_path: SPICE BSP containing Earth-centered Sun/Moon states.
        chief_spacecraft: Optional chief mass and cannonball properties.
        deputy_spacecraft: Deputy mass and cannonball properties, required for
            drag or SRP.

    Returns:
        Absolute chief/deputy histories and their equivalent RIC history.
    """
    optional_kwargs: dict[str, object] = {}
    if chief_spacecraft is not None:
        optional_kwargs["chief_spacecraft"] = chief_spacecraft
    if deputy_spacecraft is not None:
        optional_kwargs["deputy_spacecraft"] = deputy_spacecraft
    unsupported = (
        optional_kwargs.keys() - signature(_propagate_relative_numerical).parameters.keys()
    )
    if unsupported:
        raise NotImplementedError(
            "This Octavian build does not provide cannonball spacecraft "
            "properties for coupled relative propagation"
        )

    return _propagate_relative_numerical(
        chief_initial_eci,
        relative_initial_ric,
        times_s,
        deputy_initial_eci=deputy_initial_eci,
        central_body=central_body,
        perturbations=perturbations,
        initial_epoch=initial_epoch,
        max_step_s=max_step_s,
        ephemeris_step_s=ephemeris_step_s,
        bsp_path=bsp_path,
        **optional_kwargs,
    )


def relative_elements(
    initial_elements: RelativeOrbitalElements | ClassicalRelativeOrbitalElements | ArrayLike,
    times_s: ArrayLike,
    *,
    chief_initial_state_eci: BoundaryState,
    mu_m3ps2: float,
    representation: str = "damico",
    central_body: CelestialBody | str = EARTH,
    perturbations: Perturbations | None = None,
    initial_epoch: str | datetime | float | int | None = None,
    max_step_s: float = 10.0,
    ephemeris_step_s: float = 600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
    chief_spacecraft: Spacecraft | None = None,
    deputy_spacecraft: Spacecraft | None = None,
) -> RelativeElementPropagationResult:
    """Propagate relative elements and return native plus RIC histories.

    Perturbed propagation advances the coupled absolute states once. Use
    ``result.elements`` for osculating D'Amico/classical elements and
    ``result.ric`` for plotting or Cartesian relative analysis.

    Args:
        initial_elements: Initial D'Amico or classical relative elements.
        times_s: Requested elapsed output times.
        chief_initial_state_eci: Chief absolute state at elapsed time zero.
        mu_m3ps2: Central-body gravitational parameter.
        representation: ``"damico"`` or ``"classical_elements"``.
        central_body: Body constants used by perturbations.
        perturbations: Optional differential force model.
        initial_epoch: Required for Moon, Sun, or SRP.
        max_step_s: Maximum internal RK4 step.
        ephemeris_step_s: Sun/Moon interpolation spacing.
        bsp_path: SPICE BSP containing Earth-centered Sun/Moon states.
        chief_spacecraft: Optional chief mass and cannonball properties.
        deputy_spacecraft: Deputy mass and cannonball properties, required for
            drag or SRP.

    Returns:
        Paired native-element and equivalent RIC histories.
    """
    return propagate_relative_element_history(
        initial_elements,
        times_s,
        chief_initial_state_eci=chief_initial_state_eci,
        mu_m3ps2=mu_m3ps2,
        representation=representation,
        central_body=central_body,
        perturbations=perturbations,
        initial_epoch=initial_epoch,
        max_step_s=max_step_s,
        ephemeris_step_s=ephemeris_step_s,
        bsp_path=bsp_path,
        chief_spacecraft=chief_spacecraft,
        deputy_spacecraft=deputy_spacecraft,
    )


def cr3bp(
    initial_state: BoundaryState,
    times: Sequence[float],
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
    max_step: float | None = None,
) -> StateHistory:
    """Propagate a state in a circular restricted three-body system.

    Args:
        initial_state: Synodic state at ``times[0]``.
        times: Strictly monotonic SI or canonical output times.
        system: Primary-secondary CR3BP system.
        dimensional: Use dimensional SI state and time units when true.
        max_step: Maximum internal RK4 step in selected units.

    Returns:
        ``(N, 7)`` rows of ``[synodic state, time]``.
    """
    try:
        from .cislunar import propagate_cr3bp
    except ImportError as exc:
        raise ImportError("CR3BP propagation requires Octavian's cislunar module") from exc

    return propagate_cr3bp(
        initial_state,
        times,
        system=system,
        dimensional=dimensional,
        max_step=max_step,
    )


__all__ = [
    "cr3bp",
    "cwh",
    "nonlinear_ric",
    "relative",
    "relative_elements",
    "two_body",
]
