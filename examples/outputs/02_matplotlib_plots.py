"""Output example 02: save or display Matplotlib trajectory views.

The default run writes PNG files without opening a window, which also works in
notebooks and headless automation. Pass ``--show`` to open the trajectory with
the active Matplotlib GUI backend.

Run:
  python examples/outputs/02_matplotlib_plots.py
  python examples/outputs/02_matplotlib_plots.py --show
"""

from __future__ import annotations

import sys

import numpy as np

from octavian import EARTH
from octavian.viz import (
    save_trajectory_diagnostics_image,
    save_trajectory_image,
    show_trajectory,
)

radius_m = EARTH.mean_radius_m + 500_000.0
speed_mps = np.sqrt(EARTH.mu_m3ps2 / radius_m)
angles = np.linspace(0.0, 1.5 * np.pi, 240)
trajectory = np.zeros((angles.size, 7), dtype=float)
trajectory[:, 0] = radius_m * np.cos(angles)
trajectory[:, 1] = radius_m * np.sin(angles)
trajectory[:, 3] = -speed_mps * np.sin(angles)
trajectory[:, 4] = speed_mps * np.cos(angles)
trajectory[:, 6] = np.linspace(0.0, 4_200.0, angles.size)

save_trajectory_image(
    trajectory,
    "matplotlib_trajectory.png",
    title="500 km circular orbit",
)
save_trajectory_diagnostics_image(
    trajectory,
    "matplotlib_diagnostics.png",
    frame_kind="inertial",
    mu_m3ps2=EARTH.mu_m3ps2,
    title="500 km circular orbit diagnostics",
)
print("Wrote matplotlib_trajectory.png and matplotlib_diagnostics.png")

if "--show" in sys.argv:
    show_trajectory(trajectory, title="500 km circular orbit")
