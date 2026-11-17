"""Example 11e: mean J2 elements and their osculating Cartesian counterpart.

Run: python examples/composable/earth_centered/11e_mean_element_propagation.py

A mean circular orbit generally has nonzero instantaneous eccentricity.
Use a consistent transform when expressing a mean target as a Cartesian
state for a low-thrust transfer. This example compares unpowered models;
the secular propagator does not average thrust or replace the mission ODE.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from octavian import (
    EARTH,
    MeanElements,
    Perturbations,
    mean_to_osculating,
    osculating_to_mean,
    propagate,
)
from octavian.astro import cartesian_to_classic
from octavian.viz.plotly import save_trajectory_html

initial_mean = MeanElements(
    p_m=7_000_000.0,
    f=0.0,
    g=0.0,
    h=np.tan(np.deg2rad(40.0 / 2)),
    k=0.0,
    mean_longitude_rad=0.0,
)
initial = mean_to_osculating(initial_mean, central_body=EARTH)
recovered_mean = osculating_to_mean(initial, central_body=EARTH)
times_s = np.linspace(0, 2 * 86400.0, 289)
mean_history = propagate.mean_elements(recovered_mean, times_s, central_body=EARTH)
reconstructed = mean_history.to_osculating()
numerical = propagate.inertial(
    initial,
    times_s,
    central_body=EARTH,
    perturbations=Perturbations(j2=True),
    max_step_s=20.0,
)
position_error_m = np.linalg.norm(reconstructed[:, :3] - numerical[:, :3], axis=1)
elements = [
    cartesian_to_classic(r_m=row[:3], v_mps=row[3:6], mu_m3ps2=EARTH.mu_m3ps2) for row in numerical
]

figure = make_subplots(rows=3, cols=1, shared_xaxes=True)
hours = times_s / 3600
figure.add_trace(
    go.Scatter(x=hours, y=[row["a_m"] / 1e3 for row in elements], name="Osculating a [km]"),
    row=1,
    col=1,
)
figure.add_trace(
    go.Scatter(x=hours, y=np.full_like(hours, initial_mean.a_m / 1e3), name="Mean a [km]"),
    row=1,
    col=1,
)
figure.add_trace(
    go.Scatter(x=hours, y=[row["e"] for row in elements], name="Osculating e"), row=2, col=1
)
figure.add_trace(go.Scatter(x=hours, y=np.zeros_like(hours), name="Mean e"), row=2, col=1)
figure.add_trace(
    go.Scatter(x=hours, y=position_error_m / 1e3, name="Model difference [km]"), row=3, col=1
)
figure.update_xaxes(title_text="Elapsed time [h]", row=3, col=1)
figure.update_yaxes(title_text="a [km]", row=1, col=1)
figure.update_yaxes(title_text="e", row=2, col=1)
figure.update_yaxes(title_text="Position error [km]", row=3, col=1)
figure.update_layout(title="Mean circular orbit: secular J2 versus numerical J2", height=850)
figure.write_html("traj_composable_mean_elements_history.html")
save_trajectory_html(
    reconstructed,
    "traj_composable_mean_elements.html",
    title="Mean J2 orbit with short-period corrections",
)
np.savez(
    "traj_composable_mean_elements.npz",
    times_s=times_s,
    mean_elements=mean_history.elements,
    reconstructed=reconstructed,
    numerical=numerical,
)
print(f"Initial osculating e: {elements[0]['e']:.6g}; mean e: 0")
print(f"Maximum position difference over two days: {np.max(position_error_m):.1f} m")
print(
    "First-order truncation accumulates phase error; check numerical propagation for mission accuracy."
)
