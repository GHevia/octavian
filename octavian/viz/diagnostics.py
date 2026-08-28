"""Frame-aware time-series diagnostics independent of a plotting backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..astro.kepler import cartesian_to_classic
from ..cislunar import CR3BPSystem, jacobi_constant


@dataclass(frozen=True, slots=True)
class DiagnosticSeries:
    """One named quantity sampled on the trajectory time grid."""

    name: str
    values: NDArray[np.float64]
    unit: str


@dataclass(frozen=True, slots=True)
class DiagnosticPanel:
    """A related collection of quantities sharing one vertical axis."""

    title: str
    series: tuple[DiagnosticSeries, ...]
    y_axis_title: str


def relative_diagnostic_panels(
    traj: ArrayLike,
    *,
    solar_directions_ric: ArrayLike | None = None,
) -> tuple[DiagnosticPanel, ...]:
    """Build RIC state, range, and optional solar-phase time series."""
    trajectory = _trajectory(traj)
    positions = trajectory[:, 0:3]
    velocities = trajectory[:, 3:6]
    panels = [
        DiagnosticPanel(
            title="RIC position",
            series=_component_series(positions, ("R", "I", "C"), "m"),
            y_axis_title="Position (m)",
        ),
        DiagnosticPanel(
            title="RIC velocity",
            series=_component_series(
                velocities,
                ("R dot", "I dot", "C dot"),
                "m/s",
            ),
            y_axis_title="Velocity (m/s)",
        ),
        DiagnosticPanel(
            title="Relative geometry",
            series=(
                DiagnosticSeries(
                    "Range",
                    np.linalg.norm(positions, axis=1),
                    "m",
                ),
                DiagnosticSeries(
                    "Relative speed",
                    np.linalg.norm(velocities, axis=1),
                    "m/s",
                ),
            ),
            y_axis_title="Range / speed",
        ),
    ]
    if solar_directions_ric is not None:
        directions = np.asarray(solar_directions_ric, dtype=float)
        if directions.shape != positions.shape or not np.all(np.isfinite(directions)):
            raise ValueError(
                "solar_directions_ric must be finite with shape (N, 3)"
            )
        panels.append(
            DiagnosticPanel(
                title="Solar geometry",
                series=(
                    DiagnosticSeries(
                        "Solar phase angle",
                        _angles_between_deg(positions, directions),
                        "deg",
                    ),
                ),
                y_axis_title="Angle (deg)",
            )
        )
    return tuple(panels)


def inertial_diagnostic_panels(
    traj: ArrayLike,
    *,
    mu_m3ps2: float,
) -> tuple[DiagnosticPanel, ...]:
    """Build Cartesian state, radius, speed, and element time series."""
    trajectory = _trajectory(traj)
    positions = trajectory[:, 0:3]
    velocities = trajectory[:, 3:6]
    elements = [
        cartesian_to_classic(
            r_m=position,
            v_mps=velocity,
            mu_m3ps2=float(mu_m3ps2),
        )
        for position, velocity in zip(positions, velocities, strict=True)
    ]
    return (
        DiagnosticPanel(
            title="Inertial position",
            series=_component_series(positions, ("x", "y", "z"), "m"),
            y_axis_title="Position (m)",
        ),
        DiagnosticPanel(
            title="Inertial velocity",
            series=_component_series(velocities, ("vx", "vy", "vz"), "m/s"),
            y_axis_title="Velocity (m/s)",
        ),
        DiagnosticPanel(
            title="Orbit geometry",
            series=(
                DiagnosticSeries("Radius", np.linalg.norm(positions, axis=1), "m"),
                DiagnosticSeries(
                    "Speed",
                    np.linalg.norm(velocities, axis=1),
                    "m/s",
                ),
            ),
            y_axis_title="Radius / speed",
        ),
        DiagnosticPanel(
            title="Osculating elements",
            series=(
                DiagnosticSeries(
                    "Semi-major axis",
                    np.asarray([row["a_m"] for row in elements], dtype=float),
                    "m",
                ),
                DiagnosticSeries(
                    "Eccentricity",
                    np.asarray([row["e"] for row in elements], dtype=float),
                    "",
                ),
                DiagnosticSeries(
                    "Inclination",
                    np.asarray([row["inc_deg"] for row in elements], dtype=float),
                    "deg",
                ),
            ),
            y_axis_title="Element value",
        ),
    )


def cr3bp_diagnostic_panels(
    traj: ArrayLike,
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
) -> tuple[DiagnosticPanel, ...]:
    """Build dimensional or canonical CR3BP state and invariant time series."""
    trajectory = _trajectory(traj)
    positions = trajectory[:, 0:3]
    velocities = trajectory[:, 3:6]
    primary_position = (
        system.primary_position_m
        if dimensional
        else system.primary_position_nondimensional
    )
    secondary_position = (
        system.secondary_position_m
        if dimensional
        else system.secondary_position_nondimensional
    )
    primary_ranges = np.linalg.norm(
        positions - primary_position,
        axis=1,
    )
    secondary_ranges = np.linalg.norm(
        positions - secondary_position,
        axis=1,
    )
    jacobi_values = np.asarray(
        [
            jacobi_constant(row[0:6], system=system, dimensional=dimensional)
            for row in trajectory
        ],
        dtype=float,
    )
    position_unit = "m" if dimensional else "DU"
    velocity_unit = "m/s" if dimensional else "VU"
    jacobi_unit = "m²/s²" if dimensional else "canonical"
    return (
        DiagnosticPanel(
            title="Synodic position",
            series=_component_series(positions, ("x", "y", "z"), position_unit),
            y_axis_title=f"Position ({position_unit})",
        ),
        DiagnosticPanel(
            title="Synodic velocity",
            series=_component_series(
                velocities,
                ("xdot", "ydot", "zdot"),
                velocity_unit,
            ),
            y_axis_title=f"Velocity ({velocity_unit})",
        ),
        DiagnosticPanel(
            title="Primary geometry",
            series=(
                DiagnosticSeries("Primary range", primary_ranges, position_unit),
                DiagnosticSeries("Secondary range", secondary_ranges, position_unit),
                DiagnosticSeries(
                    "Synodic speed",
                    np.linalg.norm(velocities, axis=1),
                    velocity_unit,
                ),
            ),
            y_axis_title="Range / speed",
        ),
        DiagnosticPanel(
            title="CR3BP invariant",
            series=(DiagnosticSeries("Jacobi constant", jacobi_values, jacobi_unit),),
            y_axis_title=f"Jacobi constant ({jacobi_unit})",
        ),
    )


def _trajectory(traj: ArrayLike) -> NDArray[np.float64]:
    trajectory = np.asarray(traj, dtype=float)
    if (
        trajectory.ndim != 2
        or trajectory.shape[0] < 1
        or trajectory.shape[1] < 7
        or not np.all(np.isfinite(trajectory[:, 0:7]))
    ):
        raise ValueError(
            "traj must contain finite [position(3), velocity(3), time] rows"
        )
    return trajectory


def _component_series(
    values: NDArray[np.float64],
    names: tuple[str, str, str],
    unit: str,
) -> tuple[DiagnosticSeries, ...]:
    return tuple(
        DiagnosticSeries(name, values[:, index].copy(), unit)
        for index, name in enumerate(names)
    )


def _angles_between_deg(
    vectors: NDArray[np.float64],
    directions: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector_norms = np.linalg.norm(vectors, axis=1)
    direction_norms = np.linalg.norm(directions, axis=1)
    denominators = vector_norms * direction_norms
    values = np.full(vector_norms.shape, np.nan, dtype=float)
    valid = denominators > 1.0e-12
    cosines = np.sum(vectors[valid] * directions[valid], axis=1) / denominators[
        valid
    ]
    values[valid] = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return values
