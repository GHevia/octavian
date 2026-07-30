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
    for index, time_s in enumerate(times):
        propagated = propagate_two_body_state(
            initial_state,
            float(time_s),
            float(mu_m3ps2),
        )
        history[index] = np.hstack([propagated.r_m, propagated.v_mps, float(time_s)])
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
    unsupported = optional_kwargs.keys() - signature(
        _propagate_relative_numerical
    ).parameters.keys()
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
        raise ImportError(
            "CR3BP propagation requires Octavian's cislunar module"
        ) from exc

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
