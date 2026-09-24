"""Reference overlays preserve frame units, projections, and plot extents."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from octavian.cislunar import CR3BPSystem
from octavian.viz import matplotlib as mpl_viz
from octavian.viz import plotly as plotly_viz


@pytest.fixture(autouse=True)
def _close_figures():
    matplotlib.use("Agg", force=True)
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def trajectory():
    return np.asarray([[100.0, 200.0, 300.0, 0, 0, 0, 0], [400, 500, 600, 0, 0, 0, 1]])


@pytest.fixture
def reference():
    # Include negative and distant positions to exercise autoscaling.
    return {
        "name": "Reference orbit",
        "traj": [[-2e7, 3e7, -4e7], [3e7, -4e7, 5e7]],
        "color": "#F59E0B",
    }


@pytest.mark.parametrize("frame", ["inertial", "relative", "rotating"])
@pytest.mark.parametrize(
    "projection,indices", [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2)), ("3d", (0, 1, 2))]
)
def test_matplotlib_reference_units_projection_and_limits(
    trajectory, reference, frame, projection, indices
):
    builder, scale, kwargs = {
        "inertial": (mpl_viz.trajectory_figure, 0.001, {}),
        "relative": (mpl_viz.relative_trajectory_figure, 1.0, {}),
        "rotating": (mpl_viz.cr3bp_trajectory_figure, 0.001, {"system": CR3BPSystem.earth_moon()}),
    }[frame]
    figure = builder(
        trajectory, reference_trajectories=[reference], projection=projection, **kwargs
    )
    axes = figure.axes[0]
    line = next(line for line in axes.lines if line.get_label() == reference["name"])
    data = line.get_data_3d() if projection == "3d" else line.get_data()
    limits = [axes.get_xlim(), axes.get_ylim()]
    if projection == "3d":
        limits.append(axes.get_zlim())
    for values, index, (low, high) in zip(data, indices, limits, strict=True):
        assert values == pytest.approx(scale * np.asarray(reference["traj"])[:, index])
        assert low < min(values) < max(values) < high
    assert line.get_linestyle() == "--"
    assert line.get_color() == reference["color"]


@pytest.mark.parametrize("frame", ["inertial", "relative", "rotating"])
def test_html_export_contains_reference_trace(monkeypatch, tmp_path, trajectory, reference, frame):
    import plotly.graph_objects as go

    figures = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *args, **kwargs: figures.append(self))
    exporter, scale, kwargs = {
        "inertial": (plotly_viz.save_trajectory_html, 1.0, {"use_earth_texture": False}),
        "relative": (plotly_viz.save_relative_trajectory_html, 1.0, {}),
        "rotating": (
            plotly_viz.save_cr3bp_trajectory_html,
            0.001,
            {"system": CR3BPSystem.earth_moon()},
        ),
    }[frame]
    exporter(trajectory, str(tmp_path / "orbit.html"), reference_trajectories=[reference], **kwargs)
    trace = next(trace for trace in figures[0].data if trace.name == reference["name"])
    assert np.column_stack([trace.x, trace.y, trace.z]) == pytest.approx(
        scale * np.asarray(reference["traj"])
    )
    assert trace.line.dash == "dash"
    assert trace.line.color == reference["color"]


@pytest.mark.parametrize(
    "exporter", [mpl_viz.save_trajectory_image, mpl_viz.save_relative_trajectory_image]
)
def test_image_export_forwards_references(monkeypatch, tmp_path, trajectory, reference, exporter):
    figures = []
    monkeypatch.setattr(
        mpl_viz, "save_figure_image", lambda figure, *args, **kwargs: figures.append(figure)
    )
    exporter(trajectory, tmp_path / "orbit.png", reference_trajectories=[reference])
    assert reference["name"] in figures[0].axes[0].get_legend_handles_labels()[1]


@pytest.mark.parametrize(
    "builder", [mpl_viz.trajectory_figure, plotly_viz.relative_trajectory_figure]
)
@pytest.mark.parametrize("positions", [[], [[1, 2]], [[1, 2, float("nan")]]])
def test_reference_rejects_invalid_positions(trajectory, builder, positions):
    with pytest.raises(ValueError, match="finite position rows"):
        builder(trajectory, reference_trajectories=[{"traj": positions}])
