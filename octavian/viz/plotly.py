"""Plotly-based trajectory visualization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from ..cislunar import CR3BPSystem
from ..types import Maneuver
from .diagnostics import (
    cr3bp_diagnostic_panels,
    inertial_diagnostic_panels,
    relative_diagnostic_panels,
)

EARTH_RADIUS_M = 6378137.0


def _get_default_earth_texture_path() -> str:
    """Return the packaged default Earth texture path."""
    return str(files("octavian.viz.data").joinpath("cartoon_earth_map.png"))


def save_trajectory_html(
    traj: np.ndarray,
    out_html: str,
    *,
    x0_r_m: np.ndarray | None = None,
    x0_v_mps: np.ndarray | None = None,
    xf_r_m: np.ndarray | None = None,
    xf_v_mps: np.ndarray | None = None,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    title: str = "octavian trajectory",
    earth_radius_m: float = EARTH_RADIUS_M,
    use_earth_texture: bool = True,
    earth_texture_path: str | None = None,
) -> None:
    """Save a 3D Plotly HTML visualization of an ECI trajectory.

    Args:
        traj: Trajectory array with columns ``[rx, ry, rz, vx, vy, vz, t]`` in
            meters, meters per second, and seconds.
        out_html: Output HTML path.
        x0_r_m: Optional initial position override in meters.
        x0_v_mps: Optional initial velocity override in meters per second.
        xf_r_m: Optional final position override in meters.
        xf_v_mps: Optional final velocity override in meters per second.
        maneuvers: Optional maneuver markers.
        phase_segments: Optional phase interval dictionaries with ``name``,
            ``t_start_s``, ``t_end_s``, and optional ``color`` keys.
        title: Plot title.
        earth_radius_m: Earth radius used for the sphere in meters.
        use_earth_texture: Whether to render Earth with a texture map.
        earth_texture_path: Optional path to an equirectangular Earth texture.
            ``None`` uses the packaged default and ``""`` disables the texture.

    Raises:
        ValueError: If ``traj`` is not a 2D array with at least seven columns.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly visualization requires the optional viz dependencies. "
            "Install them with `pip install \"octavian[viz]\"`."
        ) from exc

    if not use_earth_texture:
        resolved_texture_path: str | None = None
    elif earth_texture_path is None:
        resolved_texture_path = _get_default_earth_texture_path()
    elif earth_texture_path == "":
        resolved_texture_path = None
    else:
        resolved_texture_path = str(Path(earth_texture_path).expanduser())

    trajectory = np.asarray(traj, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] < 7:
        raise ValueError("traj must be a 2D array with at least 7 columns: [r(3), v(3), t].")

    position_history_m = trajectory[:, 0:3]
    velocity_history_mps = trajectory[:, 3:6]
    time_history_s = trajectory[:, 6]
    final_time_s = float(time_history_s[-1])

    initial_position_m = (
        position_history_m[0].copy()
        if x0_r_m is None
        else np.asarray(x0_r_m, dtype=float).reshape(3)
    )
    initial_velocity_mps = (
        velocity_history_mps[0].copy()
        if x0_v_mps is None
        else np.asarray(x0_v_mps, dtype=float).reshape(3)
    )
    final_position_m = (
        position_history_m[-1].copy()
        if xf_r_m is None
        else np.asarray(xf_r_m, dtype=float).reshape(3)
    )
    final_velocity_mps = (
        velocity_history_mps[-1].copy()
        if xf_v_mps is None
        else np.asarray(xf_v_mps, dtype=float).reshape(3)
    )

    earth_radius = float(earth_radius_m)
    if resolved_texture_path is not None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Textured visualization requires the optional viz dependencies. "
                "Install them with `pip install \"octavian[viz]\"`."
            ) from exc

        earth_image = Image.open(resolved_texture_path).convert("RGB")
        earth_image = earth_image.resize((720, 360))
        quantized_image = earth_image.quantize(colors=256, method=Image.Quantize.MEDIANCUT).convert("P")
        palette = quantized_image.getpalette() or ([0, 0, 0] * 256)
        palette_rgb = np.array(palette, dtype=np.uint8).reshape(-1, 3)

        texture_indices = np.asarray(quantized_image, dtype=np.uint8)
        texture_height, texture_width = texture_indices.shape

        longitude = np.linspace(0.0, 2.0 * np.pi, texture_width)
        colatitude = np.linspace(0.0, np.pi, texture_height)
        longitude_grid, colatitude_grid = np.meshgrid(longitude, colatitude)

        earth_x = earth_radius * np.cos(longitude_grid) * np.sin(colatitude_grid)
        earth_y = earth_radius * np.sin(longitude_grid) * np.sin(colatitude_grid)
        earth_z = earth_radius * np.cos(colatitude_grid)

        colorscale = []
        for palette_index in range(256):
            red, green, blue = palette_rgb[palette_index]
            colorscale.append(
                [palette_index / 255.0, f"rgb({int(red)},{int(green)},{int(blue)})"]
            )

        earth_trace = go.Surface(
            x=earth_x,
            y=earth_y,
            z=earth_z,
            surfacecolor=texture_indices.astype(np.float64),
            cmin=0,
            cmax=255,
            colorscale=colorscale,
            showscale=False,
            name="Earth",
            hoverinfo="skip",
            lighting=dict(ambient=0.9, diffuse=0.8, specular=0.2, roughness=0.9),
        )
    else:
        longitude = np.linspace(0.0, 2.0 * np.pi, 60)
        colatitude = np.linspace(0.0, np.pi, 30)
        longitude_grid, colatitude_grid = np.meshgrid(longitude, colatitude)

        earth_x = earth_radius * np.cos(longitude_grid) * np.sin(colatitude_grid)
        earth_y = earth_radius * np.sin(longitude_grid) * np.sin(colatitude_grid)
        earth_z = earth_radius * np.cos(colatitude_grid)

        earth_trace = go.Surface(
            x=earth_x,
            y=earth_y,
            z=earth_z,
            showscale=False,
            colorscale=[[0.0, "blue"], [1.0, "blue"]],
            name="Earth",
            hoverinfo="skip",
        )

    trajectory_trace = go.Scatter3d(
        x=position_history_m[:, 0],
        y=position_history_m[:, 1],
        z=position_history_m[:, 2],
        mode="lines",
        name="Trajectory outline",
        line=dict(width=3, color="rgba(255,255,255,0.35)"),
    )

    phase_traces = []
    for phase_index, segment in enumerate(phase_segments or (), start=1):
        t_start_s = float(segment["t_start_s"])
        t_end_s = float(segment["t_end_s"])
        phase_mask = (time_history_s >= t_start_s - 1.0e-9) & (time_history_s <= t_end_s + 1.0e-9)
        if int(np.count_nonzero(phase_mask)) < 2:
            continue
        phase_positions_m = position_history_m[phase_mask]
        phase_name = str(segment.get("name", f"phase {phase_index}"))
        phase_color = str(segment.get("color", "white"))
        phase_traces.append(
            go.Scatter3d(
                x=phase_positions_m[:, 0],
                y=phase_positions_m[:, 1],
                z=phase_positions_m[:, 2],
                mode="lines",
                name=phase_name,
                line=dict(width=7, color=phase_color),
            )
        )

    start_hover_text = (
        f"<b>Start</b><br>"
        f"t = {time_history_s[0]:.3f} s<br>"
        f"r0 = [{initial_position_m[0]:.3f}, {initial_position_m[1]:.3f}, {initial_position_m[2]:.3f}] m<br>"
        f"v0 = [{initial_velocity_mps[0]:.6f}, {initial_velocity_mps[1]:.6f}, {initial_velocity_mps[2]:.6f}] m/s<br>"
        f"<b>TOF</b> = {final_time_s:.3f} s"
    )
    end_hover_text = (
        f"<b>End</b><br>"
        f"t = {final_time_s:.3f} s<br>"
        f"rf = [{final_position_m[0]:.3f}, {final_position_m[1]:.3f}, {final_position_m[2]:.3f}] m<br>"
        f"vf = [{final_velocity_mps[0]:.6f}, {final_velocity_mps[1]:.6f}, {final_velocity_mps[2]:.6f}] m/s<br>"
        f"<b>TOF</b> = {final_time_s:.3f} s"
    )

    start_trace = go.Scatter3d(
        x=[position_history_m[0, 0]],
        y=[position_history_m[0, 1]],
        z=[position_history_m[0, 2]],
        mode="markers",
        marker=dict(size=6, color="white"),
        name="Start",
        text=[start_hover_text],
        hovertemplate="%{text}<extra></extra>",
    )
    end_trace = go.Scatter3d(
        x=[position_history_m[-1, 0]],
        y=[position_history_m[-1, 1]],
        z=[position_history_m[-1, 2]],
        mode="markers",
        marker=dict(size=6, color="white"),
        name="End",
        text=[end_hover_text],
        hovertemplate="%{text}<extra></extra>",
    )

    maneuver_traces = []
    for maneuver_index, maneuver in enumerate(maneuvers or (), start=1):
        maneuver_position_m = np.asarray(maneuver.r_m, dtype=float).reshape(3)
        maneuver_time_s = float(maneuver.t_s)
        maneuver_delta_v_mps = np.asarray(maneuver.dv_mps, dtype=float).reshape(3)
        maneuver_delta_v_mag_mps = float(np.linalg.norm(maneuver_delta_v_mps))
        maneuver_hover_text = (
            f"<b>{maneuver.name}</b><br>"
            f"t = {maneuver_time_s:.3f} s<br>"
            f"r = [{maneuver_position_m[0]:.3f}, {maneuver_position_m[1]:.3f}, {maneuver_position_m[2]:.3f}] m<br>"
            f"Δv = [{maneuver_delta_v_mps[0]:.6f}, {maneuver_delta_v_mps[1]:.6f}, {maneuver_delta_v_mps[2]:.6f}] m/s<br>"
            f"|Δv| = {maneuver_delta_v_mag_mps:.6f} m/s"
        )
        maneuver_traces.append(
            go.Scatter3d(
                x=[maneuver_position_m[0]],
                y=[maneuver_position_m[1]],
                z=[maneuver_position_m[2]],
                mode="markers",
                marker=dict(size=7, color="red"),
                name=f"M{maneuver_index}: {maneuver.name}",
                text=[maneuver_hover_text],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    figure = go.Figure(
        data=[earth_trace, trajectory_trace, *phase_traces, start_trace, end_trace, *maneuver_traces]
    )
    figure.update_layout(
        title=dict(text=title, x=0.5),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        scene=dict(
            bgcolor="black",
            xaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            yaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            zaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
            ),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    )
    figure.write_html(out_html, include_plotlyjs="cdn")


def cr3bp_trajectory_figure(
    traj: np.ndarray,
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    reference_trajectories: Sequence[dict[str, object]] | None = None,
    title: str = "octavian CR3BP trajectory",
) -> Any:
    """Build an interactive barycentric-synodic CR3BP trajectory figure.

    Args:
        traj: Rows ``[x, y, z, xdot, ydot, zdot, time]``.
        system: Primary-secondary CR3BP system.
        dimensional: Interpret positions as meters when true or canonical
            distance units otherwise.
        maneuvers: Optional maneuver markers in the trajectory's units.
        phase_segments: Optional phase interval dictionaries with ``name``,
            ``t_start_s``, ``t_end_s``, and optional ``color`` keys.
        reference_trajectories: Optional dictionaries with ``name`` and
            ``traj`` keys plus an optional Plotly ``color``. These are useful
            for plotting departure and arrival periodic orbits around a
            solved transfer.
        title: Figure title.

    Returns:
        A Plotly 3D figure with both bodies and all five Lagrange points.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly visualization requires the optional viz dependencies. "
            "Install them with `pip install \"octavian[viz]\"`."
        ) from exc

    trajectory = np.asarray(traj, dtype=float)
    if (
        trajectory.ndim != 2
        or trajectory.shape[0] < 1
        or trajectory.shape[1] < 7
        or not np.all(np.isfinite(trajectory[:, 0:7]))
    ):
        raise ValueError("traj must contain finite [x, y, z, xdot, ydot, zdot, time] rows")
    scale = 1.0 / 1_000.0 if dimensional else 1.0
    unit = "km" if dimensional else "DU"
    primary_position = (
        system.primary_position_m if dimensional else system.primary_position_nondimensional
    )
    secondary_position = (
        system.secondary_position_m if dimensional else system.secondary_position_nondimensional
    )
    lagrange_points = system.lagrange_points(dimensional=dimensional)
    positions = scale * trajectory[:, 0:3]
    hover_text = [
        (
            f"t = {row[6]:.3f}<br>"
            f"x,y,z = [{position[0]:.6f}, {position[1]:.6f}, "
            f"{position[2]:.6f}] {unit}"
        )
        for row, position in zip(trajectory, positions, strict=True)
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines",
            name="Trajectory",
            line=dict(width=6, color="#00CC96"),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    for reference_index, reference in enumerate(reference_trajectories or (), start=1):
        reference_rows = np.asarray(reference["traj"], dtype=float)
        if (
            reference_rows.ndim != 2
            or reference_rows.shape[0] < 1
            or reference_rows.shape[1] < 3
            or not np.all(np.isfinite(reference_rows[:, 0:3]))
        ):
            raise ValueError("Each CR3BP reference trajectory must contain finite position rows")
        reference_positions = scale * reference_rows[:, 0:3]
        figure.add_trace(
            go.Scatter3d(
                x=reference_positions[:, 0],
                y=reference_positions[:, 1],
                z=reference_positions[:, 2],
                mode="lines",
                name=str(reference.get("name", f"Reference {reference_index}")),
                line=dict(
                    width=3,
                    color=str(reference.get("color", "#A0AEC0")),
                    dash="dash",
                ),
                opacity=0.75,
            )
        )
    for phase_index, segment in enumerate(phase_segments or (), start=1):
        start_time = float(segment["t_start_s"])
        end_time = float(segment["t_end_s"])
        phase_mask = (trajectory[:, 6] >= start_time - 1.0e-9) & (
            trajectory[:, 6] <= end_time + 1.0e-9
        )
        if int(np.count_nonzero(phase_mask)) < 2:
            continue
        phase_positions = positions[phase_mask]
        figure.add_trace(
            go.Scatter3d(
                x=phase_positions[:, 0],
                y=phase_positions[:, 1],
                z=phase_positions[:, 2],
                mode="lines",
                name=str(segment.get("name", f"Phase {phase_index}")),
                line=dict(
                    width=8,
                    color=str(segment.get("color", "#00CC96")),
                ),
            )
        )
    for maneuver_index, maneuver in enumerate(maneuvers or (), start=1):
        maneuver_position = scale * np.asarray(maneuver.r_m, dtype=float).reshape(3)
        delta_v = np.asarray(maneuver.dv_mps, dtype=float).reshape(3)
        figure.add_trace(
            go.Scatter3d(
                x=[maneuver_position[0]],
                y=[maneuver_position[1]],
                z=[maneuver_position[2]],
                mode="markers",
                name=f"M{maneuver_index}: {maneuver.name}",
                marker=dict(size=7, color="#FFA15A", symbol="diamond"),
                text=[
                    (
                        f"<b>{maneuver.name}</b><br>"
                        f"t = {float(maneuver.t_s):.3f}<br>"
                        f"|Δv| = {float(np.linalg.norm(delta_v)):.6f}"
                    )
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter3d(
            x=[scale * primary_position[0], scale * secondary_position[0]],
            y=[0.0, 0.0],
            z=[0.0, 0.0],
            mode="markers+text",
            name="Primaries",
            text=[system.primary.name.title(), system.secondary.name.title()],
            textposition="top center",
            marker=dict(size=[16, 9], color=["#636EFA", "#AB63FA"]),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[scale * point[0] for point in lagrange_points.values()],
            y=[scale * point[1] for point in lagrange_points.values()],
            z=[scale * point[2] for point in lagrange_points.values()],
            mode="markers+text",
            name="Lagrange points",
            text=list(lagrange_points),
            textposition="top center",
            marker=dict(size=5, color="#EF553B", symbol="diamond"),
        )
    )
    axis_style = dict(
        backgroundcolor="black",
        gridcolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.25)",
    )
    figure.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title=f"Synodic X ({unit})", **axis_style),
            yaxis=dict(title=f"Synodic Y ({unit})", **axis_style),
            zaxis=dict(title=f"Synodic Z ({unit})", **axis_style),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return figure


def save_cr3bp_trajectory_html(
    traj: np.ndarray,
    out_html: str,
    *,
    system: CR3BPSystem,
    dimensional: bool = True,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    reference_trajectories: Sequence[dict[str, object]] | None = None,
    title: str = "octavian CR3BP trajectory",
) -> None:
    """Save a barycentric-synodic CR3BP trajectory as interactive HTML.

    Args:
        traj: Rows ``[x, y, z, xdot, ydot, zdot, time]``.
        out_html: Destination HTML path.
        system: Primary-secondary CR3BP system.
        dimensional: Interpret trajectory and marker positions as meters when
            true or canonical distance units otherwise.
        maneuvers: Optional maneuver markers in the trajectory's units.
        phase_segments: Optional phase interval dictionaries with ``name``,
            ``t_start_s``, ``t_end_s``, and optional ``color`` keys.
        reference_trajectories: Optional named trajectory dictionaries to
            overlay, such as departure and arrival periodic orbits.
        title: Figure title.
    """
    figure = cr3bp_trajectory_figure(
        traj,
        system=system,
        dimensional=dimensional,
        maneuvers=maneuvers,
        phase_segments=phase_segments,
        reference_trajectories=reference_trajectories,
        title=title,
    )
    figure.write_html(out_html, include_plotlyjs="cdn")


def relative_trajectory_figure(
    traj: np.ndarray,
    *,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    title: str = "octavian relative trajectory",
    chief_radius_m: float = 0.0,
) -> Any:
    """Build an interactive 3D RIC trajectory figure.

    Args:
        traj: Relative trajectory rows ``[R, I, C, Rdot, Idot, Cdot, t]`` in
            meters, meters per second, and seconds.
        maneuvers: Optional maneuver markers expressed in the same RIC frame.
        phase_segments: Optional phase interval dictionaries with ``name``,
            ``t_start_s``, ``t_end_s``, and optional ``color`` keys.
        title: Plot title.
        chief_radius_m: Optional physical or keep-out radius drawn about the
            chief.  Zero draws a marker without a surrounding sphere.

    Returns:
        A :class:`plotly.graph_objects.Figure` with equal RIC axis scaling.

    Raises:
        ValueError: If the trajectory shape, values, or chief radius is invalid.
        RuntimeError: If Plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly visualization requires the optional viz dependencies. "
            "Install them with `pip install \"octavian[viz]\"`."
        ) from exc

    trajectory = np.asarray(traj, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] < 7:
        raise ValueError(
            "traj must be a 2D array with at least 7 columns: "
            "[R, I, C, Rdot, Idot, Cdot, t]."
        )
    if trajectory.shape[0] < 1 or not np.all(np.isfinite(trajectory[:, 0:7])):
        raise ValueError("traj must contain at least one finite RIC state")
    chief_radius = float(chief_radius_m)
    if not np.isfinite(chief_radius) or chief_radius < 0.0:
        raise ValueError("chief_radius_m must be finite and non-negative")

    position_history_m = trajectory[:, 0:3]
    velocity_history_mps = trajectory[:, 3:6]
    time_history_s = trajectory[:, 6]
    distance_history_m = np.linalg.norm(position_history_m, axis=1)
    speed_history_mps = np.linalg.norm(velocity_history_mps, axis=1)
    hover_text = [
        (
            f"t = {time_s:.3f} s<br>"
            f"RIC = [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m<br>"
            f"|ρ| = {distance:.3f} m<br>"
            f"|ρ̇| = {speed:.6f} m/s"
        )
        for position, distance, speed, time_s in zip(
            position_history_m,
            distance_history_m,
            speed_history_mps,
            time_history_s,
            strict=True,
        )
    ]

    traces: list[Any] = [
        go.Scatter3d(
            x=position_history_m[:, 0],
            y=position_history_m[:, 1],
            z=position_history_m[:, 2],
            mode="lines",
            name="Relative trajectory",
            line=dict(width=6, color="#3BA3FF"),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    ]

    for phase_index, segment in enumerate(phase_segments or (), start=1):
        start_s = float(segment["t_start_s"])
        end_s = float(segment["t_end_s"])
        mask = (time_history_s >= start_s - 1.0e-9) & (
            time_history_s <= end_s + 1.0e-9
        )
        if int(np.count_nonzero(mask)) < 2:
            continue
        positions = position_history_m[mask]
        traces.append(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="lines",
                name=str(segment.get("name", f"phase {phase_index}")),
                line=dict(width=8, color=str(segment.get("color", "#3BA3FF"))),
            )
        )

    traces.extend(
        [
            go.Scatter3d(
                x=[position_history_m[0, 0]],
                y=[position_history_m[0, 1]],
                z=[position_history_m[0, 2]],
                mode="markers",
                name="Start",
                marker=dict(size=6, color="#69DB7C"),
                text=[hover_text[0]],
                hovertemplate="%{text}<extra></extra>",
            ),
            go.Scatter3d(
                x=[position_history_m[-1, 0]],
                y=[position_history_m[-1, 1]],
                z=[position_history_m[-1, 2]],
                mode="markers",
                name="End",
                marker=dict(size=6, color="#FFD43B"),
                text=[hover_text[-1]],
                hovertemplate="%{text}<extra></extra>",
            ),
            go.Scatter3d(
                x=[0.0],
                y=[0.0],
                z=[0.0],
                mode="markers",
                name="Chief",
                marker=dict(size=8, color="white", symbol="diamond"),
                hovertemplate="<b>Chief / RIC origin</b><extra></extra>",
            ),
        ]
    )

    if chief_radius > 0.0:
        longitude = np.linspace(0.0, 2.0 * np.pi, 48)
        colatitude = np.linspace(0.0, np.pi, 24)
        longitude_grid, colatitude_grid = np.meshgrid(longitude, colatitude)
        traces.insert(
            0,
            go.Surface(
                x=chief_radius * np.cos(longitude_grid) * np.sin(colatitude_grid),
                y=chief_radius * np.sin(longitude_grid) * np.sin(colatitude_grid),
                z=chief_radius * np.cos(colatitude_grid),
                colorscale=[[0.0, "#868E96"], [1.0, "#ADB5BD"]],
                opacity=0.55,
                showscale=False,
                name="Chief boundary",
                hovertemplate=f"Chief boundary: {chief_radius:.3f} m<extra></extra>",
            ),
        )

    for index, maneuver in enumerate(maneuvers or (), start=1):
        position = np.asarray(maneuver.r_m, dtype=float).reshape(3)
        delta_v = np.asarray(maneuver.dv_mps, dtype=float).reshape(3)
        text = (
            f"<b>{maneuver.name}</b><br>"
            f"t = {float(maneuver.t_s):.3f} s<br>"
            f"RIC = [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] m<br>"
            f"Δv = [{delta_v[0]:.6f}, {delta_v[1]:.6f}, {delta_v[2]:.6f}] m/s<br>"
            f"|Δv| = {float(np.linalg.norm(delta_v)):.6f} m/s"
        )
        traces.append(
            go.Scatter3d(
                x=[position[0]],
                y=[position[1]],
                z=[position[2]],
                mode="markers",
                name=f"M{index}: {maneuver.name}",
                marker=dict(size=7, color="#FF6B6B"),
                text=[text],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    axis_style = dict(
        backgroundcolor="black",
        gridcolor="rgba(255,255,255,0.12)",
        zerolinecolor="rgba(255,255,255,0.35)",
        showspikes=False,
    )
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=dict(text=title, x=0.5),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        scene=dict(
            bgcolor="black",
            xaxis={**axis_style, "title": "Radial, R (m)"},
            yaxis={**axis_style, "title": "In-track, I (m)"},
            zaxis={**axis_style, "title": "Cross-track, C (m)"},
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    )
    return figure


def save_relative_trajectory_html(
    traj: np.ndarray,
    out_html: str,
    *,
    maneuvers: Sequence[Maneuver] | None = None,
    phase_segments: Sequence[dict[str, object]] | None = None,
    title: str = "octavian relative trajectory",
    chief_radius_m: float = 0.0,
) -> None:
    """Save an interactive chief-centered RIC trajectory as HTML.

    This is the relative-motion counterpart to :func:`save_trajectory_html`.
    It labels the RIC axes explicitly, places the chief at the origin, and does
    not draw Earth in a frame where that geometry would be misleading.
    """
    figure = relative_trajectory_figure(
        traj,
        maneuvers=maneuvers,
        phase_segments=phase_segments,
        title=title,
        chief_radius_m=chief_radius_m,
    )
    figure.write_html(out_html, include_plotlyjs="cdn")


def trajectory_diagnostics_figure(
    traj: np.ndarray,
    *,
    frame_kind: str,
    mu_m3ps2: float | None = None,
    solar_directions_ric: np.ndarray | None = None,
    cr3bp_system: CR3BPSystem | None = None,
    cr3bp_dimensional: bool = True,
    title: str = "octavian trajectory diagnostics",
) -> Any:
    """Build stacked, shared-time plots appropriate to the trajectory frame."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError(
            "Plotly visualization requires the optional viz dependencies. "
            "Install them with `pip install \"octavian[viz]\"`."
        ) from exc

    trajectory = np.asarray(traj, dtype=float)
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
            raise ValueError(
                "Inertial trajectory diagnostics require a positive mu_m3ps2"
            )
        panels = inertial_diagnostic_panels(
            trajectory,
            mu_m3ps2=float(mu_m3ps2),
        )

    figure = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.08, 0.25 / len(panels)),
        subplot_titles=[panel.title for panel in panels],
    )
    time_values = trajectory[:, 6]
    for row, panel in enumerate(panels, start=1):
        for series in panel.series:
            label = (
                f"{series.name} ({series.unit})" if series.unit else series.name
            )
            figure.add_trace(
                go.Scatter(
                    x=time_values,
                    y=series.values,
                    mode="lines",
                    name=label,
                ),
                row=row,
                col=1,
            )
        figure.update_yaxes(title_text=panel.y_axis_title, row=row, col=1)
    time_unit = (
        "TU"
        if normalized_frame == "rotating" and not cr3bp_dimensional
        else "s"
    )
    figure.update_xaxes(
        title_text=f"Time ({time_unit})",
        row=len(panels),
        col=1,
    )
    figure.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_dark",
        height=max(650, 245 * len(panels)),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(l=75, r=25, t=90, b=55),
    )
    return figure


def save_trajectory_diagnostics_html(
    traj: np.ndarray,
    out_html: str,
    *,
    frame_kind: str,
    mu_m3ps2: float | None = None,
    solar_directions_ric: np.ndarray | None = None,
    cr3bp_system: CR3BPSystem | None = None,
    cr3bp_dimensional: bool = True,
    title: str = "octavian trajectory diagnostics",
) -> None:
    """Save frame-aware trajectory time histories as interactive HTML."""
    figure = trajectory_diagnostics_figure(
        traj,
        frame_kind=frame_kind,
        mu_m3ps2=mu_m3ps2,
        solar_directions_ric=solar_directions_ric,
        cr3bp_system=cr3bp_system,
        cr3bp_dimensional=cr3bp_dimensional,
        title=title,
    )
    figure.write_html(out_html, include_plotlyjs="cdn")
