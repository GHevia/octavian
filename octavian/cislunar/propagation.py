"""Numerical propagation and invariants for CR3BP trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..specs import BoundaryState
from .model import CR3BPSystem
from .transforms import dimensionalize_state, nondimensionalize_state

StateHistory = NDArray[np.float64]


def cr3bp_derivative(
    state: ArrayLike,
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
) -> NDArray[np.float64]:
    """Return a six-state CR3BP derivative in synodic coordinates.

    Args:
        state: ``[x, y, z, xdot, ydot, zdot]``.
        system: Physical primary-secondary system.
        dimensional: Interpret and return SI units when true; canonical CR3BP
            units otherwise.
    """
    values = np.asarray(state, dtype=float).reshape(6)
    if dimensional:
        canonical = nondimensionalize_state(
            BoundaryState(values[0:3], values[3:6]),
            system,
        )
        derivative_canonical = _canonical_derivative(
            np.hstack([canonical.r_m, canonical.v_mps]),
            system.mass_parameter,
        )
        return np.hstack(
            [
                derivative_canonical[0:3] * system.velocity_scale_mps,
                derivative_canonical[3:6] * system.separation_m / system.time_scale_s**2,
            ]
        )
    return _canonical_derivative(values, system.mass_parameter)


def jacobi_constant(
    state: ArrayLike,
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
) -> float:
    """Return the Jacobi integral for one synodic CR3BP state."""
    values = np.asarray(state, dtype=float).reshape(6)
    if dimensional:
        canonical = nondimensionalize_state(
            BoundaryState(values[0:3], values[3:6]),
            system,
        )
        canonical_value = _canonical_jacobi(
            np.hstack([canonical.r_m, canonical.v_mps]),
            system.mass_parameter,
        )
        return canonical_value * system.velocity_scale_mps**2
    return _canonical_jacobi(values, system.mass_parameter)


def propagate_cr3bp(
    initial_state: BoundaryState,
    times: Sequence[float],
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
    max_step: float | None = None,
) -> StateHistory:
    """Propagate a CR3BP state with deterministic fourth-order Runge-Kutta.

    Requested times must be strictly monotonic and may run forward or
    backward. The returned columns are ``[x, y, z, xdot, ydot, zdot, time]``.

    Args:
        initial_state: State at ``times[0]``.
        times: Output times in seconds or canonical time.
        system: Physical primary-secondary system.
        dimensional: Use SI state/time units when true.
        max_step: Maximum internal step in the selected time units. Defaults
            to 900 seconds or 0.0025 canonical time.
    """
    output_times = np.asarray(times, dtype=float).reshape(-1)
    if output_times.size < 2:
        raise ValueError("propagate_cr3bp requires at least two output times")
    differences = np.diff(output_times)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError("CR3BP output times must be strictly monotonic")
    if not np.all(np.isfinite(output_times)):
        raise ValueError("CR3BP output times must be finite")
    initial_values = np.hstack([initial_state.r_m, initial_state.v_mps])
    if not np.all(np.isfinite(initial_values)):
        raise ValueError("CR3BP initial state must be finite")
    step_limit = float(max_step) if max_step is not None else (900.0 if dimensional else 0.0025)
    if not np.isfinite(step_limit) or step_limit <= 0.0:
        raise ValueError("max_step must be finite and positive")

    if dimensional:
        canonical_state = nondimensionalize_state(initial_state, system)
        state = np.hstack([canonical_state.r_m, canonical_state.v_mps])
        canonical_times = output_times / system.time_scale_s
        canonical_step = step_limit / system.time_scale_s
    else:
        state = initial_values.astype(float)
        canonical_times = output_times
        canonical_step = step_limit

    history = np.empty((output_times.size, 7), dtype=float)
    history[0, 0:6] = np.hstack([initial_state.r_m, initial_state.v_mps])
    history[0, 6] = output_times[0]
    current_time = float(canonical_times[0])
    for output_index in range(1, output_times.size):
        target_time = float(canonical_times[output_index])
        interval = target_time - current_time
        steps = max(1, int(np.ceil(abs(interval) / canonical_step)))
        step = interval / steps
        for _ in range(steps):
            state = _rk4_step(
                state,
                step,
                system.mass_parameter,
            )
            current_time += step
        if dimensional:
            dimensional_state = dimensionalize_state(
                BoundaryState(state[0:3], state[3:6]),
                system,
            )
            history[output_index, 0:6] = np.hstack([dimensional_state.r_m, dimensional_state.v_mps])
        else:
            history[output_index, 0:6] = state
        history[output_index, 6] = output_times[output_index]
    return history


def _canonical_derivative(state: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
    """Return canonical CR3BP equations of motion."""
    x, y, z, xdot, ydot, zdot = state
    primary_distance = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    secondary_distance = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)
    if primary_distance <= 0.0 or secondary_distance <= 0.0:
        raise ValueError("CR3BP state lies at a primary or secondary singularity")
    xddot = (
        2.0 * ydot
        + x
        - (1.0 - mu) * (x + mu) / primary_distance**3
        - mu * (x - 1.0 + mu) / secondary_distance**3
    )
    yddot = -2.0 * xdot + y - (1.0 - mu) * y / primary_distance**3 - mu * y / secondary_distance**3
    zddot = -(1.0 - mu) * z / primary_distance**3 - mu * z / secondary_distance**3
    return np.asarray([xdot, ydot, zdot, xddot, yddot, zddot], dtype=float)


def _canonical_jacobi(state: NDArray[np.float64], mu: float) -> float:
    """Return canonical Jacobi constant."""
    x, y, z, xdot, ydot, zdot = state
    primary_distance = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    secondary_distance = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)
    potential_twice = (
        x**2 + y**2 + 2.0 * (1.0 - mu) / primary_distance + 2.0 * mu / secondary_distance
    )
    return float(potential_twice - xdot**2 - ydot**2 - zdot**2)


def _rk4_step(
    state: NDArray[np.float64],
    step: float,
    mu: float,
) -> NDArray[np.float64]:
    """Advance one canonical fourth-order Runge-Kutta step."""
    k1 = _canonical_derivative(state, mu)
    k2 = _canonical_derivative(state + 0.5 * step * k1, mu)
    k3 = _canonical_derivative(state + 0.5 * step * k2, mu)
    k4 = _canonical_derivative(state + step * k3, mu)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
