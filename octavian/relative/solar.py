"""SPICE-driven Sun directions for chief-centered RIC geometry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..data.ephemeris import DEFAULT_EPHEMERIS_BSP, sample_sun_moon_positions_eci_tod
from ..specs import BoundaryState
from .transforms import ric_basis


@dataclass(frozen=True, slots=True)
class SolarDirectionTable:
    """Sampled Sun geometry for relative constraints and diagnostics."""

    times_s: NDArray[np.float64]
    directions_ric: NDArray[np.float64]
    sun_positions_eci_m: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=float).reshape(-1)
        directions = np.asarray(self.directions_ric, dtype=float)
        if times.size < 2 or directions.shape != (times.size, 3):
            raise ValueError("SolarDirectionTable requires N>=2 times and an (N, 3) array")
        if np.any(np.diff(times) <= 0.0) or not np.all(np.isfinite(directions)):
            raise ValueError("SolarDirectionTable samples must be finite and increasing")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("SolarDirectionTable directions must have non-zero norm")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "directions_ric", directions / norms[:, None])
        if self.sun_positions_eci_m is not None:
            positions = np.asarray(self.sun_positions_eci_m, dtype=float)
            if positions.shape != (times.size, 3) or not np.all(np.isfinite(positions)):
                raise ValueError(
                    "SolarDirectionTable sun positions must have shape (N, 3)"
                )
            object.__setattr__(self, "sun_positions_eci_m", positions)

    def at(self, times_s: ArrayLike) -> NDArray[np.float64]:
        """Interpolate and renormalize Sun directions at elapsed times."""
        query = np.asarray(times_s, dtype=float)
        if np.any(query < self.times_s[0]) or np.any(query > self.times_s[-1]):
            raise ValueError("Requested solar direction lies outside the sampled time range")
        flat_query = query.reshape(-1)
        interpolated = np.column_stack(
            [
                np.interp(flat_query, self.times_s, self.directions_ric[:, component])
                for component in range(3)
            ]
        )
        norms = np.linalg.norm(interpolated, axis=1)
        result = interpolated / norms[:, None]
        return result.reshape((*query.shape, 3))

    def sun_position_at(self, times_s: ArrayLike) -> NDArray[np.float64]:
        """Interpolate the SPICE-derived Sun position in ECI."""
        if self.sun_positions_eci_m is None:
            raise ValueError("SolarDirectionTable does not contain ECI Sun positions")
        query = np.asarray(times_s, dtype=float)
        if np.any(query < self.times_s[0]) or np.any(query > self.times_s[-1]):
            raise ValueError("Requested Sun position lies outside the sampled time range")
        flat_query = query.reshape(-1)
        interpolated = np.column_stack(
            [
                np.interp(
                    flat_query,
                    self.times_s,
                    self.sun_positions_eci_m[:, component],
                )
                for component in range(3)
            ]
        )
        return interpolated.reshape((*query.shape, 3))


def circular_chief_state(
    initial_state_eci: BoundaryState,
    elapsed_time_s: float,
    mean_motion_radps: float,
) -> BoundaryState:
    """Propagate a circular chief reference state analytically."""
    angle = float(mean_motion_radps) * float(elapsed_time_s)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    position = (
        initial_state_eci.r_m * cosine
        + initial_state_eci.v_mps * (sine / float(mean_motion_radps))
    )
    velocity = (
        -initial_state_eci.r_m * (float(mean_motion_radps) * sine)
        + initial_state_eci.v_mps * cosine
    )
    return BoundaryState(position, velocity)


def sample_solar_directions_ric(
    *,
    chief_initial_state_eci: BoundaryState,
    mean_motion_radps: float,
    initial_epoch: str | datetime | float | int,
    duration_s: float,
    step_s: float = 600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
) -> SolarDirectionTable:
    """Sample the SPICE Sun line and rotate it into the chief's RIC frame.

    The chief state and bundled BSP both use Octavian's Earth-centered
    ``ECI_TOD`` orientation for this calculation.
    """
    if float(mean_motion_radps) <= 0.0:
        raise ValueError("mean_motion_radps must be positive")
    times_s, body_positions = sample_sun_moon_positions_eci_tod(
        initial_epoch=initial_epoch,
        duration_s=float(duration_s),
        step_s=float(step_s),
        bsp_path=bsp_path,
    )
    directions = np.empty((times_s.size, 3), dtype=float)
    for index, (time_s, sun_position_m) in enumerate(
        zip(times_s, body_positions["sun"], strict=True)
    ):
        chief = circular_chief_state(
            chief_initial_state_eci,
            float(time_s),
            float(mean_motion_radps),
        )
        sun_line_eci = sun_position_m - chief.r_m
        sun_line_ric = ric_basis(chief.r_m, chief.v_mps) @ sun_line_eci
        directions[index] = sun_line_ric / np.linalg.norm(sun_line_ric)
    return SolarDirectionTable(
        times_s=times_s,
        directions_ric=directions,
        sun_positions_eci_m=body_positions["sun"],
    )
