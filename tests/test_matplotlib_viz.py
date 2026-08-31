from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from PIL import Image

from octavian.cislunar import CR3BPSystem
from octavian.solution import Solution
from octavian.solvers.preconfigured import RendezvousResult
from octavian.viz.matplotlib import (
    cr3bp_trajectory_figure,
    relative_trajectory_figure,
    save_trajectory_image,
    show_trajectory,
    trajectory_diagnostics_figure,
    trajectory_figure,
)


@pytest.fixture(autouse=True)
def _headless_matplotlib():
    matplotlib.use("Agg", force=True)
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _inertial_trajectory() -> np.ndarray:
    radius_m = 7_000_000.0
    mu_m3ps2 = 3.986004418e14
    speed_mps = np.sqrt(mu_m3ps2 / radius_m)
    angles = np.linspace(0.0, 0.7 * np.pi, 24)
    trajectory = np.zeros((angles.size, 7), dtype=float)
    trajectory[:, 0] = radius_m * np.cos(angles)
    trajectory[:, 1] = radius_m * np.sin(angles)
    trajectory[:, 3] = -speed_mps * np.sin(angles)
    trajectory[:, 4] = speed_mps * np.cos(angles)
    trajectory[:, 6] = np.linspace(0.0, 2_000.0, angles.size)
    return trajectory


def test_inertial_figure_labels_axes_and_earth() -> None:
    figure = trajectory_figure(_inertial_trajectory(), title="Static orbit")
    axes = figure.axes[0]

    assert axes.get_title() == "Static orbit"
    assert axes.get_xlabel() == "ECI X (km)"
    assert axes.get_ylabel() == "ECI Y (km)"
    assert axes.get_zlabel() == "ECI Z (km)"
    assert {line.get_label() for line in axes.lines} >= {"Trajectory"}
    assert {"Earth", "Trajectory", "Start", "End"} <= set(axes.get_legend_handles_labels()[1])


@pytest.mark.parametrize(
    ("suffix", "expected_format"),
    [(".png", "PNG"), (".jpg", "JPEG")],
)
def test_save_trajectory_image_writes_png_and_jpeg(
    tmp_path: Path,
    suffix: str,
    expected_format: str,
) -> None:
    output = tmp_path / f"trajectory{suffix}"

    save_trajectory_image(_inertial_trajectory(), output, dpi=72)

    with Image.open(output) as image:
        assert image.format == expected_format
        assert image.width > 100
        assert image.height > 100


def test_save_trajectory_image_rejects_non_image_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in"):
        save_trajectory_image(_inertial_trajectory(), tmp_path / "trajectory.svg")


def test_relative_figure_labels_ric_axes_and_chief() -> None:
    trajectory = np.asarray(
        [
            [100.0, -500.0, 20.0, 0.1, 0.0, 0.0, 0.0],
            [50.0, -100.0, 5.0, 0.0, 0.1, 0.0, 100.0],
        ]
    )

    figure = relative_trajectory_figure(trajectory, chief_radius_m=10.0)
    axes = figure.axes[0]

    assert axes.get_xlabel() == "Radial, R (m)"
    assert axes.get_ylabel() == "In-track, I (m)"
    assert axes.get_zlabel() == "Cross-track, C (m)"
    assert "Chief" in axes.get_legend_handles_labels()[1]


def test_cr3bp_figure_selects_lagrange_points_and_labels_bodies() -> None:
    system = CR3BPSystem.earth_moon()
    trajectory = np.zeros((2, 7), dtype=float)
    trajectory[:, 6] = [0.0, 1.0]

    figure = cr3bp_trajectory_figure(
        trajectory,
        system=system,
        dimensional=False,
        lagrange_point_names=("l1", "L2"),
    )
    labels = figure.axes[0].get_legend_handles_labels()[1]

    assert labels == ["Trajectory", "Earth", "Moon", "L1", "L2"]
    with pytest.raises(ValueError, match="must be L1"):
        cr3bp_trajectory_figure(
            trajectory,
            system=system,
            lagrange_point_names=("L6",),
        )


def test_diagnostics_figure_uses_shared_time_panels() -> None:
    trajectory = np.asarray(
        [
            [100.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [50.0, 25.0, 0.0, 0.0, 0.1, 0.0, 10.0],
        ]
    )

    figure = trajectory_diagnostics_figure(trajectory, frame_kind="relative")

    assert len(figure.axes) == 3
    assert figure.axes[-1].get_xlabel() == "Time (s)"
    assert figure.axes[0].get_shared_x_axes().joined(figure.axes[0], figure.axes[-1])


def test_show_trajectory_uses_matplotlib_gui_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.pyplot as plt

    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))

    figure = show_trajectory(_inertial_trajectory())

    assert calls == ["show"]
    assert figure.axes[0].get_xlabel() == "ECI X (km)"


def test_solution_viz_saves_frame_aware_image(tmp_path: Path) -> None:
    result = RendezvousResult(
        converged=True,
        traj=_inertial_trajectory(),
        maneuvers=[],
        info={"mu_m3ps2": 3.986004418e14},
    )
    solution = Solution(ok=True, result=result)
    output = tmp_path / "solution.png"

    solution.viz().save_image(output, dpi=72)

    assert output.is_file()
    assert solution.viz().figure().axes[0].get_xlabel() == "ECI X (km)"
    assert len(solution.viz().diagnostics_figure().axes) == 4
