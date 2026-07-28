from __future__ import annotations

import numpy as np
import pytest

from octavian.viz.plotly import relative_trajectory_figure

pytest.importorskip("plotly")


def test_relative_trajectory_figure_labels_ric_axes_and_chief() -> None:
    trajectory = np.asarray(
        [
            [100.0, -500.0, 20.0, 0.1, 0.0, 0.0, 0.0],
            [50.0, -100.0, 5.0, 0.0, 0.1, 0.0, 100.0],
        ]
    )
    figure = relative_trajectory_figure(
        trajectory,
        chief_radius_m=10.0,
    )

    assert figure.layout.scene.xaxis.title.text == "Radial, R (m)"
    assert figure.layout.scene.yaxis.title.text == "In-track, I (m)"
    assert figure.layout.scene.zaxis.title.text == "Cross-track, C (m)"
    assert "Chief" in [trace.name for trace in figure.data]
    assert "Relative trajectory" in [trace.name for trace in figure.data]
