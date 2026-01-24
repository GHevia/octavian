from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from ..types import Maneuver

EARTH_RADIUS_M = 6378137.0

def save_trajectory_html(
    traj: np.ndarray,
    out_html: str,
    *,
    x0_r_m: np.ndarray,
    x0_v_mps: np.ndarray,
    xf_r_m: np.ndarray,
    xf_v_mps: np.ndarray,
    maneuvers: Optional[Sequence[Maneuver]] = None,
    title: str = "octavian trajectory",
    earth_radius_m: float = EARTH_RADIUS_M,
) -> None:
    """Save a 3D Plotly HTML visualization of an ECI trajectory."""
    import plotly.graph_objects as go

    traj = np.asarray(traj, float)
    r = traj[:, 0:3]
    t = traj[:, 6]
    tf = float(t[-1])

    x0_r = np.asarray(x0_r_m, float).reshape(3)
    x0_v = np.asarray(x0_v_mps, float).reshape(3)
    xf_r = np.asarray(xf_r_m, float).reshape(3)
    xf_v = np.asarray(xf_v_mps, float).reshape(3)

    # Earth sphere
    nu, nv = 60, 30
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)
    Re = float(earth_radius_m)
    xs = Re * np.cos(uu) * np.sin(vv)
    ys = Re * np.sin(uu) * np.sin(vv)
    zs = Re * np.cos(vv)

    earth = go.Surface(
        x=xs, y=ys, z=zs,
        showscale=False,
        colorscale=[[0.0, "blue"], [1.0, "blue"]],
        name="Earth",
        text="Earth (spherical)",
        hoverinfo="text"
        # opacity=1,
    )

    line = go.Scatter3d(
        x=r[:, 0], y=r[:, 1], z=r[:, 2],
        mode="lines",
        name="Trajectory",
        line=dict(width=6, color="white"),
        # hoverinfo="skip",
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
        x=[r[0, 0]], y=[r[0, 1]], z=[r[0, 2]],
        mode="markers",
        marker=dict(size=6, color="white"),
        name="Start",
        text=[start_text],
        hovertemplate="%{text}<extra></extra>",
    )

    end = go.Scatter3d(
        x=[r[-1, 0]], y=[r[-1, 1]], z=[r[-1, 2]],
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
                    x=[mr[0]], y=[mr[1]], z=[mr[2]],
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
            xaxis=dict(backgroundcolor="black", gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.2)"),
            yaxis=dict(backgroundcolor="black", gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.2)"),
            zaxis=dict(backgroundcolor="black", gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.2)"),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    )
    fig.write_html(out_html, include_plotlyjs="cdn")
