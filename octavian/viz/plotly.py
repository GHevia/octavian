from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path

import numpy as np

from ..types import Maneuver

EARTH_RADIUS_M = 6378137.0


def _get_default_earth_texture_path() -> str:
    """Return the packaged default Earth texture path."""
    return str(files("octavian.viz.data").joinpath("cartoon_earth_map.PNG"))


def save_trajectory_html(
    traj: np.ndarray,
    out_html: str,
    *,
    x0_r_m: np.ndarray | None = None,
    x0_v_mps: np.ndarray | None = None,
    xf_r_m: np.ndarray | None = None,
    xf_v_mps: np.ndarray | None = None,
    maneuvers: Sequence[Maneuver] | None = None,
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
        x0_r_m: Optional initial position override in meters. If omitted, the
            first trajectory position is used.
        x0_v_mps: Optional initial velocity override in meters per second. If
            omitted, the first trajectory velocity is used.
        xf_r_m: Optional final position override in meters. If omitted, the
            last trajectory position is used.
        xf_v_mps: Optional final velocity override in meters per second. If
            omitted, the last trajectory velocity is used.
        maneuvers: Optional maneuver markers.
        title: Plot title.
        earth_radius_m: Earth radius used for the sphere in meters.
        use_earth_texture: Whether to render Earth with a texture map.
        earth_texture_path: Optional path to an equirectangular Earth texture.
            ``None`` uses the packaged default, ``""`` disables the texture,
            and any other value is resolved as a filesystem path.

    Raises:
        ValueError: If ``traj`` is not a 2D array with at least seven columns.
    """
    import plotly.graph_objects as go

    # -----------------------------
    # Resolve texture path / toggle
    # -----------------------------
    if not use_earth_texture:
        resolved_texture_path: str | None = None
    else:
        if earth_texture_path is None:
            resolved_texture_path = _get_default_earth_texture_path()
        elif earth_texture_path == "":
            resolved_texture_path = None
        else:
            resolved_texture_path = str(Path(earth_texture_path).expanduser())

    # -----------------------------
    # Normalize inputs
    # -----------------------------
    traj = np.asarray(traj, float)
    if traj.ndim != 2 or traj.shape[1] < 7:
        raise ValueError("traj must be a 2D array with at least 7 columns: [r(3), v(3), t].")

    r = traj[:, 0:3]
    v = traj[:, 3:6]
    t = traj[:, 6]
    tf = float(t[-1])

    # Infer endpoints from trajectory unless overrides provided
    x0_r = r[0].copy() if x0_r_m is None else np.asarray(x0_r_m, float).reshape(3)

    x0_v = v[0].copy() if x0_v_mps is None else np.asarray(x0_v_mps, float).reshape(3)

    xf_r = r[-1].copy() if xf_r_m is None else np.asarray(xf_r_m, float).reshape(3)

    xf_v = v[-1].copy() if xf_v_mps is None else np.asarray(xf_v_mps, float).reshape(3)

    # -----------------------------
    # Resolve texture path / toggle
    # -----------------------------
    if not use_earth_texture:
        resolved_texture_path: str | None = None
    else:
        if earth_texture_path is None:
            resolved_texture_path = _get_default_earth_texture_path()
        elif earth_texture_path == "":
            resolved_texture_path = None
        else:
            resolved_texture_path = str(Path(earth_texture_path).expanduser())

    # -----------------------------
    # Normalize inputs
    # -----------------------------
    traj = np.asarray(traj, float)
    if traj.ndim != 2 or traj.shape[1] < 7:
        raise ValueError("traj must be a 2D array with at least 7 columns: [r(3), v(3), t].")

    r = traj[:, 0:3]
    t = traj[:, 6]
    tf = float(t[-1])

    # breakpoint()

    # x0_r = np.asarray(x0_r_m, float).reshape(3)
    # x0_v = np.asarray(x0_v_mps, float).reshape(3)
    # xf_r = np.asarray(xf_r_m, float).reshape(3)
    # xf_v = np.asarray(xf_v_mps, float).reshape(3)

    # -----------------------------
    # Earth sphere (optionally textured)
    # -----------------------------
    Re = float(earth_radius_m)

    if resolved_texture_path is not None:
        from PIL import Image

        img = Image.open(resolved_texture_path).convert("RGB")
        img = img.resize((720, 360))  # keep modest

        # Quantize to a small palette so we can build a small colorscale (<=256 stops)
        img_q = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT).convert("P")
        palette = img_q.getpalette() or ([0, 0, 0] * 256)
        pal = np.array(palette, dtype=np.uint8).reshape(-1, 3)  # (256, 3)

        tex_idx = np.asarray(img_q, dtype=np.uint8)  # (H, W)
        H, W = tex_idx.shape

        u = np.linspace(0.0, 2.0 * np.pi, W)
        v = np.linspace(0.0, np.pi, H)
        uu, vv = np.meshgrid(u, v)

        xs = Re * np.cos(uu) * np.sin(vv)
        ys = Re * np.sin(uu) * np.sin(vv)
        zs = Re * np.cos(vv)

        colorscale = []
        for k in range(256):
            rr, gg, bb = pal[k]
            s = k / 255.0
            colorscale.append([s, f"rgb({int(rr)},{int(gg)},{int(bb)})"])

        earth = go.Surface(
            x=xs,
            y=ys,
            z=zs,
            surfacecolor=tex_idx.astype(np.float64),  # 0..255
            cmin=0,
            cmax=255,
            colorscale=colorscale,
            showscale=False,
            name="Earth",
            hoverinfo="skip",
            lighting=dict(ambient=0.9, diffuse=0.8, specular=0.2, roughness=0.9),
        )
    else:
        nu, nv = 60, 30
        u = np.linspace(0, 2 * np.pi, nu)
        v = np.linspace(0, np.pi, nv)
        uu, vv = np.meshgrid(u, v)

        xs = Re * np.cos(uu) * np.sin(vv)
        ys = Re * np.sin(uu) * np.sin(vv)
        zs = Re * np.cos(vv)

        earth = go.Surface(
            x=xs,
            y=ys,
            z=zs,
            showscale=False,
            colorscale=[[0.0, "blue"], [1.0, "blue"]],
            name="Earth",
            hoverinfo="skip",
        )

    # -----------------------------
    # Trajectory + markers
    # -----------------------------
    line = go.Scatter3d(
        x=r[:, 0],
        y=r[:, 1],
        z=r[:, 2],
        mode="lines",
        name="Trajectory",
        line=dict(width=6, color="white"),
    )

    start_text = (
        f"<b>Start</b><br>"
        f"t = {t[0]:.3f} s<br>"
        f"r0 = [{x0_r[0]:.3f}, {x0_r[1]:.3f}, {x0_r[2]:.3f}] m<br>"
        f"v0 = [{x0_v[0]:.6f}, {x0_v[1]:.6f}, {x0_v[2]:.6f}] m/s<br>"
        f"<b>TOF</b> = {tf:.3f} s"
    )

    end_text = (
        f"<b>End</b><br>"
        f"t = {tf:.3f} s<br>"
        f"rf = [{xf_r[0]:.3f}, {xf_r[1]:.3f}, {xf_r[2]:.3f}] m<br>"
        f"vf = [{xf_v[0]:.6f}, {xf_v[1]:.6f}, {xf_v[2]:.6f}] m/s<br>"
        f"<b>TOF</b> = {tf:.3f} s"
    )

    start = go.Scatter3d(
        x=[r[0, 0]],
        y=[r[0, 1]],
        z=[r[0, 2]],
        mode="markers",
        marker=dict(size=6, color="white"),
        name="Start",
        text=[start_text],
        hovertemplate="%{text}<extra></extra>",
    )

    end = go.Scatter3d(
        x=[r[-1, 0]],
        y=[r[-1, 1]],
        z=[r[-1, 2]],
        mode="markers",
        marker=dict(size=6, color="white"),
        name="End",
        text=[end_text],
        hovertemplate="%{text}<extra></extra>",
    )

    man_traces = []
    if maneuvers is not None:
        for i, m in enumerate(maneuvers, start=1):
            mr = np.asarray(m.r_m, float).reshape(3)
            mt = float(m.t_s)
            mdv = np.asarray(m.dv_mps, float).reshape(3)
            mdv_mag = float(np.linalg.norm(mdv))
            text = (
                f"<b>{m.name}</b><br>"
                f"t = {mt:.3f} s<br>"
                f"r = [{mr[0]:.3f}, {mr[1]:.3f}, {mr[2]:.3f}] m<br>"
                f"Δv = [{mdv[0]:.6f}, {mdv[1]:.6f}, {mdv[2]:.6f}] m/s<br>"
                f"|Δv| = {mdv_mag:.6f} m/s"
            )
            man_traces.append(
                go.Scatter3d(
                    x=[mr[0]],
                    y=[mr[1]],
                    z=[mr[2]],
                    mode="markers",
                    marker=dict(size=7, color="red"),
                    name=f"M{i}: {m.name}",
                    text=[text],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    fig = go.Figure(data=[earth, line, start, end, *man_traces])
    fig.update_layout(
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
    fig.write_html(out_html, include_plotlyjs="cdn")
