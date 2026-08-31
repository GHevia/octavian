"""Matplotlib-based trajectory and diagnostic visualization helpers.

The functions in this module mirror Octavian's Plotly views while producing
ordinary Matplotlib figures. Figure builders support further customization,
``save_*_image`` helpers write PNG or JPEG files, and ``show_*`` helpers open
the active Matplotlib GUI backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..cislunar import CR3BPSystem
from ..types import Maneuver
from .constants import EARTH_RADIUS_M
from .diagnostics import (
    cr3bp_diagnostic_panels,
    inertial_diagnostic_panels,
    relative_diagnostic_panels,
)

_BACKGROUND_COLOR = "#101418"
_FOREGROUND_COLOR = "#F1F3F5"
_GRID_COLOR = "#495057"
_TRAJECTORY_COLOR = "#22B8CF"


def trajectory_figure(
    traj: np.ndarray,
    *,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    title: str = "octavian trajectory",
    earth_radius_m: float = EARTH_RADIUS_M,
    figsize: tuple[float, float] = (9.0, 7.0),
) -> Any:
    """Build a static 3D Earth-centered inertial trajectory figure.

    Args:
        traj: Rows ``[rx, ry, rz, vx, vy, vz, time]`` in SI units.
        maneuvers: Optional maneuver markers.
        phase_segments: Optional phase dictionaries with ``name``,
            ``t_start_s``, ``t_end_s``, and optional ``color`` keys.
        title: Figure title.
        earth_radius_m: Earth radius in meters.
        figsize: Matplotlib figure size in inches.

    Returns:
        A :class:`matplotlib.figure.Figure`. Positions are displayed in km.
    """
    plt = _pyplot()
    trajectory = _trajectory(traj)
    earth_radius = float(earth_radius_m)
    if not np.isfinite(earth_radius) or earth_radius <= 0.0:
        raise ValueError("earth_radius_m must be finite and positive")

    positions_km = trajectory[:, 0:3] / 1_000.0
    time_s = trajectory[:, 6]
    figure = plt.figure(figsize=figsize)
    axes = figure.add_subplot(111, projection="3d")
    _style_3d_axes(figure, axes, title)

    longitude = np.linspace(0.0, 2.0 * np.pi, 64)
    colatitude = np.linspace(0.0, np.pi, 32)
    longitude_grid, colatitude_grid = np.meshgrid(longitude, colatitude)
    earth_radius_km = earth_radius / 1_000.0
    axes.plot_surface(
        earth_radius_km * np.cos(longitude_grid) * np.sin(colatitude_grid),
        earth_radius_km * np.sin(longitude_grid) * np.sin(colatitude_grid),
        earth_radius_km * np.cos(colatitude_grid),
        color="#3B82F6",
        alpha=0.7,
        linewidth=0.0,
        shade=True,
    )
    axes.scatter([], [], [], color="#3B82F6", s=70, label="Earth")
    axes.plot(
        positions_km[:, 0],
        positions_km[:, 1],
        positions_km[:, 2],
        color=_TRAJECTORY_COLOR,
        linewidth=2.2,
        label="Trajectory",
    )
    _plot_phase_segments(axes, positions_km, time_s, phase_segments)
    axes.scatter(
        *positions_km[0],
        color="#69DB7C",
        s=40,
        label="Start",
        depthshade=False,
    )
    axes.scatter(
        *positions_km[-1],
        color="#FFD43B",
        s=40,
        label="End",
        depthshade=False,
    )
    for index, maneuver in enumerate(maneuvers or (), start=1):
        position_km = np.asarray(maneuver.r_m, dtype=float).reshape(3) / 1_000.0
        axes.scatter(
            *position_km,
            color="#FF6B6B",
            marker="D",
            s=48,
            label=f"M{index}: {maneuver.name}",
            depthshade=False,
        )

    axes.set_xlabel("ECI X (km)")
    axes.set_ylabel("ECI Y (km)")
    axes.set_zlabel("ECI Z (km)")
    _finish_3d_figure(figure, axes)
    return figure


def relative_trajectory_figure(
    traj: np.ndarray,
    *,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    title: str = "octavian relative trajectory",
    chief_radius_m: float = 0.0,
    figsize: tuple[float, float] = (9.0, 7.0),
) -> Any:
    """Build a static 3D chief-centered RIC trajectory figure."""
    plt = _pyplot()
    trajectory = _trajectory(traj)
    chief_radius = float(chief_radius_m)
    if not np.isfinite(chief_radius) or chief_radius < 0.0:
        raise ValueError("chief_radius_m must be finite and non-negative")

    positions = trajectory[:, 0:3]
    time_s = trajectory[:, 6]
    figure = plt.figure(figsize=figsize)
    axes = figure.add_subplot(111, projection="3d")
    _style_3d_axes(figure, axes, title)
    axes.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="#3BA3FF",
        linewidth=2.2,
        label="Relative trajectory",
    )
    _plot_phase_segments(axes, positions, time_s, phase_segments)
    axes.scatter(
        *positions[0],
        color="#69DB7C",
        s=40,
        label="Start",
        depthshade=False,
    )
    axes.scatter(
        *positions[-1],
        color="#FFD43B",
        s=40,
        label="End",
        depthshade=False,
    )
    axes.scatter(
        0.0,
        0.0,
        0.0,
        color="#F8F9FA",
        marker="D",
        s=55,
        label="Chief",
        depthshade=False,
    )
    if chief_radius > 0.0:
        longitude = np.linspace(0.0, 2.0 * np.pi, 48)
        colatitude = np.linspace(0.0, np.pi, 24)
        longitude_grid, colatitude_grid = np.meshgrid(longitude, colatitude)
        axes.plot_surface(
            chief_radius * np.cos(longitude_grid) * np.sin(colatitude_grid),
            chief_radius * np.sin(longitude_grid) * np.sin(colatitude_grid),
            chief_radius * np.cos(colatitude_grid),
            color="#868E96",
            alpha=0.45,
            linewidth=0.0,
        )
    for index, maneuver in enumerate(maneuvers or (), start=1):
        position = np.asarray(maneuver.r_m, dtype=float).reshape(3)
        axes.scatter(
            *position,
            color="#FF6B6B",
            marker="D",
            s=48,
            label=f"M{index}: {maneuver.name}",
            depthshade=False,
        )

    axes.set_xlabel("Radial, R (m)")
    axes.set_ylabel("In-track, I (m)")
    axes.set_zlabel("Cross-track, C (m)")
    _finish_3d_figure(figure, axes)
    return figure


def cr3bp_trajectory_figure(
    traj: np.ndarray,
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
    lagrange_point_names: Sequence[str] | None = None,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    reference_trajectories: Sequence[dict[str, object]] | None = None,
    title: str = "octavian CR3BP trajectory",
    figsize: tuple[float, float] = (9.0, 7.0),
) -> Any:
    """Build a static barycentric-synodic CR3BP trajectory figure."""
    plt = _pyplot()
    trajectory = _trajectory(traj)
    scale = 1.0 / 1_000.0 if dimensional else 1.0
    unit = "km" if dimensional else "DU"
    positions = scale * trajectory[:, 0:3]
    time_values = trajectory[:, 6]
    primary_position = scale * (
        system.primary_position_m if dimensional else system.primary_position_nondimensional
    )
    secondary_position = scale * (
        system.secondary_position_m if dimensional else system.secondary_position_nondimensional
    )
    all_lagrange_points = system.lagrange_points(dimensional=dimensional)
    selected_lagrange_names = _selected_lagrange_names(
        all_lagrange_points,
        lagrange_point_names,
    )

    figure = plt.figure(figsize=figsize)
    axes = figure.add_subplot(111, projection="3d")
    _style_3d_axes(figure, axes, title)
    axes.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="#00CC96",
        linewidth=2.4,
        label="Trajectory",
    )
    for index, reference in enumerate(reference_trajectories or (), start=1):
        reference_rows = _position_rows(reference.get("traj"))
        reference_positions = scale * reference_rows[:, 0:3]
        axes.plot(
            reference_positions[:, 0],
            reference_positions[:, 1],
            reference_positions[:, 2],
            color=str(reference.get("color", "#A0AEC0")),
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
            label=str(reference.get("name", f"Reference {index}")),
        )
    _plot_phase_segments(axes, positions, time_values, phase_segments)
    for index, maneuver in enumerate(maneuvers or (), start=1):
        position = scale * np.asarray(maneuver.r_m, dtype=float).reshape(3)
        axes.scatter(
            *position,
            color="#FFA15A",
            marker="D",
            s=48,
            label=f"M{index}: {maneuver.name}",
            depthshade=False,
        )
    for body_name, body_position, marker_size, marker_color in (
        (system.primary.name.title(), primary_position, 130, "#3B82F6"),
        (system.secondary.name.title(), secondary_position, 75, "#9CA3AF"),
    ):
        axes.scatter(
            *body_position,
            color=marker_color,
            s=marker_size,
            label=body_name,
            depthshade=False,
        )
        axes.text(*body_position, f" {body_name}", color=_FOREGROUND_COLOR)
    for point_name in selected_lagrange_names:
        point = scale * all_lagrange_points[point_name]
        axes.scatter(
            *point,
            color="#EF553B",
            marker="D",
            s=32,
            label=point_name,
            depthshade=False,
        )
        axes.text(*point, f" {point_name}", color=_FOREGROUND_COLOR)

    axes.set_xlabel(f"Synodic X ({unit})")
    axes.set_ylabel(f"Synodic Y ({unit})")
    axes.set_zlabel(f"Synodic Z ({unit})")
    _finish_3d_figure(figure, axes)
    return figure


def trajectory_diagnostics_figure(
    traj: np.ndarray,
    *,
    frame_kind: str,
    mu_m3ps2: float | None = None,
    solar_directions_ric: np.ndarray | None = None,
    cr3bp_system: CR3BPSystem | None = None,
    cr3bp_dimensional: bool = True,
    title: str = "octavian trajectory diagnostics",
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Build static, shared-time diagnostic panels for a trajectory frame."""
    plt = _pyplot()
    trajectory = _trajectory(traj)
    normalized_frame = str(frame_kind).strip().lower()
    if normalized_frame == "relative":
        panels = relative_diagnostic_panels(
            trajectory,
            solar_directions_ric=solar_directions_ric,
        )
    elif normalized_frame == "rotating":
        if cr3bp_system is None:
            raise ValueError("Rotating trajectory diagnostics require cr3bp_system")
        panels = cr3bp_diagnostic_panels(
            trajectory,
            system=cr3bp_system,
            dimensional=cr3bp_dimensional,
        )
    else:
        if mu_m3ps2 is None or float(mu_m3ps2) <= 0.0:
            raise ValueError("Inertial trajectory diagnostics require a positive mu_m3ps2")
        panels = inertial_diagnostic_panels(
            trajectory,
            mu_m3ps2=float(mu_m3ps2),
        )

    resolved_figsize = figsize or (10.0, max(6.0, 2.35 * len(panels)))
    figure, axes_values = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=resolved_figsize,
        squeeze=False,
    )
    axes_list = axes_values[:, 0]
    figure.patch.set_facecolor(_BACKGROUND_COLOR)
    time_values = trajectory[:, 6]
    for axes, panel in zip(axes_list, panels, strict=True):
        _style_2d_axes(axes)
        for series in panel.series:
            label = f"{series.name} ({series.unit})" if series.unit else series.name
            axes.plot(time_values, series.values, linewidth=1.7, label=label)
        axes.set_title(panel.title, color=_FOREGROUND_COLOR)
        axes.set_ylabel(panel.y_axis_title, color=_FOREGROUND_COLOR)
        axes.legend(
            loc="best",
            facecolor=_BACKGROUND_COLOR,
            edgecolor=_GRID_COLOR,
            labelcolor=_FOREGROUND_COLOR,
        )
    time_unit = "TU" if normalized_frame == "rotating" and not cr3bp_dimensional else "s"
    axes_list[-1].set_xlabel(f"Time ({time_unit})", color=_FOREGROUND_COLOR)
    figure.suptitle(title, color=_FOREGROUND_COLOR, fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def save_trajectory_image(
    traj: np.ndarray,
    out_image: str | Path,
    *,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    title: str = "octavian trajectory",
    earth_radius_m: float = EARTH_RADIUS_M,
    figsize: tuple[float, float] = (9.0, 7.0),
    dpi: int = 160,
) -> None:
    """Save an Earth-centered trajectory as PNG or JPEG."""
    figure = trajectory_figure(
        traj,
        maneuvers=maneuvers,
        phase_segments=phase_segments,
        title=title,
        earth_radius_m=earth_radius_m,
        figsize=figsize,
    )
    save_figure_image(figure, out_image, dpi=dpi)


def save_relative_trajectory_image(
    traj: np.ndarray,
    out_image: str | Path,
    *,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    title: str = "octavian relative trajectory",
    chief_radius_m: float = 0.0,
    figsize: tuple[float, float] = (9.0, 7.0),
    dpi: int = 160,
) -> None:
    """Save a chief-centered RIC trajectory as PNG or JPEG."""
    figure = relative_trajectory_figure(
        traj,
        maneuvers=maneuvers,
        phase_segments=phase_segments,
        title=title,
        chief_radius_m=chief_radius_m,
        figsize=figsize,
    )
    save_figure_image(figure, out_image, dpi=dpi)


def save_cr3bp_trajectory_image(
    traj: np.ndarray,
    out_image: str | Path,
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
    lagrange_point_names: Sequence[str] | None = None,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    reference_trajectories: Sequence[dict[str, object]] | None = None,
    title: str = "octavian CR3BP trajectory",
    figsize: tuple[float, float] = (9.0, 7.0),
    dpi: int = 160,
) -> None:
    """Save a barycentric-synodic CR3BP trajectory as PNG or JPEG."""
    figure = cr3bp_trajectory_figure(
        traj,
        system=system,
        dimensional=dimensional,
        lagrange_point_names=lagrange_point_names,
        maneuvers=maneuvers,
        phase_segments=phase_segments,
        reference_trajectories=reference_trajectories,
        title=title,
        figsize=figsize,
    )
    save_figure_image(figure, out_image, dpi=dpi)


def save_trajectory_diagnostics_image(
    traj: np.ndarray,
    out_image: str | Path,
    *,
    frame_kind: str,
    mu_m3ps2: float | None = None,
    solar_directions_ric: np.ndarray | None = None,
    cr3bp_system: CR3BPSystem | None = None,
    cr3bp_dimensional: bool = True,
    title: str = "octavian trajectory diagnostics",
    figsize: tuple[float, float] | None = None,
    dpi: int = 160,
) -> None:
    """Save frame-aware trajectory diagnostic panels as PNG or JPEG."""
    figure = trajectory_diagnostics_figure(
        traj,
        frame_kind=frame_kind,
        mu_m3ps2=mu_m3ps2,
        solar_directions_ric=solar_directions_ric,
        cr3bp_system=cr3bp_system,
        cr3bp_dimensional=cr3bp_dimensional,
        title=title,
        figsize=figsize,
    )
    save_figure_image(figure, out_image, dpi=dpi)


def show_trajectory(traj: np.ndarray, **kwargs: Any) -> Any:
    """Open an Earth-centered trajectory with the active GUI backend."""
    return show_figure(trajectory_figure(traj, **kwargs))


def show_relative_trajectory(traj: np.ndarray, **kwargs: Any) -> Any:
    """Open a chief-centered RIC trajectory with the active GUI backend."""
    return show_figure(relative_trajectory_figure(traj, **kwargs))


def show_cr3bp_trajectory(traj: np.ndarray, **kwargs: Any) -> Any:
    """Open a barycentric-synodic trajectory with the active GUI backend."""
    return show_figure(cr3bp_trajectory_figure(traj, **kwargs))


def show_trajectory_diagnostics(traj: np.ndarray, **kwargs: Any) -> Any:
    """Open frame-aware trajectory diagnostics with the active GUI backend."""
    return show_figure(trajectory_diagnostics_figure(traj, **kwargs))


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib visualization requires the optional viz dependencies. "
            'Install them with `pip install "octavian[viz]"`.'
        ) from exc
    return plt


def _trajectory(traj: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(traj, dtype=float)
    if (
        trajectory.ndim != 2
        or trajectory.shape[0] < 1
        or trajectory.shape[1] < 7
        or not np.all(np.isfinite(trajectory[:, 0:7]))
    ):
        raise ValueError("traj must contain finite [position(3), velocity(3), time] rows")
    return trajectory


def _position_rows(value: object) -> np.ndarray:
    rows = np.asarray(value, dtype=float)
    if (
        rows.ndim != 2
        or rows.shape[0] < 1
        or rows.shape[1] < 3
        or not np.all(np.isfinite(rows[:, 0:3]))
    ):
        raise ValueError("Each reference trajectory must contain finite position rows")
    return rows


def _selected_lagrange_names(
    all_lagrange_points: dict[str, np.ndarray],
    names: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = (
        tuple(all_lagrange_points)
        if names is None
        else tuple(str(name).strip().upper() for name in names)
    )
    unknown = [name for name in selected if name not in all_lagrange_points]
    if unknown:
        raise ValueError(
            f"lagrange_point_names entries must be L1, L2, L3, L4, or L5; received {unknown!r}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("lagrange_point_names entries must be unique")
    return selected


def _plot_phase_segments(
    axes: Any,
    positions: np.ndarray,
    time_values: np.ndarray,
    phase_segments: Sequence[dict[str, object]] | None,
) -> None:
    for index, segment in enumerate(phase_segments or (), start=1):
        start_time = float(segment["t_start_s"])
        end_time = float(segment["t_end_s"])
        mask = (time_values >= start_time - 1.0e-9) & (time_values <= end_time + 1.0e-9)
        if int(np.count_nonzero(mask)) < 2:
            continue
        phase_positions = positions[mask]
        axes.plot(
            phase_positions[:, 0],
            phase_positions[:, 1],
            phase_positions[:, 2],
            color=str(segment.get("color", _TRAJECTORY_COLOR)),
            linewidth=3.2,
            label=str(segment.get("name", f"Phase {index}")),
        )


def _style_3d_axes(figure: Any, axes: Any, title: str) -> None:
    figure.patch.set_facecolor(_BACKGROUND_COLOR)
    axes.set_facecolor(_BACKGROUND_COLOR)
    axes.set_title(title, color=_FOREGROUND_COLOR, pad=16)
    axes.tick_params(colors=_FOREGROUND_COLOR)
    axes.xaxis.label.set_color(_FOREGROUND_COLOR)
    axes.yaxis.label.set_color(_FOREGROUND_COLOR)
    axes.zaxis.label.set_color(_FOREGROUND_COLOR)
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.pane.set_facecolor(_BACKGROUND_COLOR)
        axis.pane.set_edgecolor(_GRID_COLOR)
        axis._axinfo["grid"]["color"] = _GRID_COLOR


def _style_2d_axes(axes: Any) -> None:
    axes.set_facecolor(_BACKGROUND_COLOR)
    axes.tick_params(colors=_FOREGROUND_COLOR)
    for spine in axes.spines.values():
        spine.set_color(_GRID_COLOR)
    axes.grid(True, color=_GRID_COLOR, alpha=0.45, linewidth=0.7)


def _finish_3d_figure(figure: Any, axes: Any) -> None:
    _set_axes_equal(axes)
    legend = axes.legend(
        loc="best",
        facecolor=_BACKGROUND_COLOR,
        edgecolor=_GRID_COLOR,
        labelcolor=_FOREGROUND_COLOR,
    )
    legend.get_frame().set_alpha(0.85)
    figure.tight_layout()


def _set_axes_equal(axes: Any) -> None:
    limits = np.asarray(
        [axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d()],
        dtype=float,
    )
    centers = np.mean(limits, axis=1)
    radius = 0.5 * float(np.max(np.ptp(limits, axis=1)))
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    axes.set_xlim3d(centers[0] - radius, centers[0] + radius)
    axes.set_ylim3d(centers[1] - radius, centers[1] + radius)
    axes.set_zlim3d(centers[2] - radius, centers[2] + radius)
    axes.set_box_aspect((1.0, 1.0, 1.0))


def save_figure_image(
    figure: Any,
    out_image: str | Path,
    *,
    dpi: int = 160,
) -> None:
    """Save an existing Matplotlib figure as PNG or JPEG, then close it."""
    plt = _pyplot()
    try:
        path = Path(out_image).expanduser()
        suffix = path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("out_image must end in .png, .jpg, or .jpeg")
        resolved_dpi = int(dpi)
        if resolved_dpi <= 0:
            raise ValueError("dpi must be positive")
        figure.savefig(
            path,
            dpi=resolved_dpi,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
    finally:
        plt.close(figure)


def show_figure(figure: Any) -> Any:
    """Display an existing figure with the active Matplotlib GUI backend."""
    _pyplot().show()
    return figure


__all__ = [
    "cr3bp_trajectory_figure",
    "relative_trajectory_figure",
    "save_figure_image",
    "save_cr3bp_trajectory_image",
    "save_relative_trajectory_image",
    "save_trajectory_diagnostics_image",
    "save_trajectory_image",
    "show_cr3bp_trajectory",
    "show_figure",
    "show_relative_trajectory",
    "show_trajectory",
    "show_trajectory_diagnostics",
    "trajectory_diagnostics_figure",
    "trajectory_figure",
]
