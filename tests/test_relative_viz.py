from __future__ import annotations

import numpy as np
import pytest

from octavian.viz.diagnostics import (
    inertial_diagnostic_panels,
    relative_diagnostic_panels,
)
from octavian.viz.plotly import (
    relative_trajectory_figure,
    trajectory_diagnostics_figure,
)

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


def test_relative_diagnostics_include_range_and_solar_phase() -> None:
    trajectory = np.asarray(
        [
            [100.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 200.0, 0.0, 0.0, 0.2, 0.0, 10.0],
        ]
    )
    sun_directions = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    panels = relative_diagnostic_panels(
        trajectory,
        solar_directions_ric=sun_directions,
    )

    assert [panel.title for panel in panels] == [
        "RIC position",
        "RIC velocity",
        "Relative geometry",
        "Solar geometry",
    ]
    assert panels[2].series[0].values == pytest.approx([100.0, 200.0])
    assert panels[3].series[0].values == pytest.approx([0.0, 90.0])


def test_inertial_diagnostics_include_radius_and_orbital_elements() -> None:
    mu = 3.986004418e14
    radius = 7_000_000.0
    speed = np.sqrt(mu / radius)
    trajectory = np.asarray(
        [
            [radius, 0.0, 0.0, 0.0, speed, 0.0, 0.0],
            [0.0, radius, 0.0, -speed, 0.0, 0.0, 100.0],
        ]
    )

    panels = inertial_diagnostic_panels(trajectory, mu_m3ps2=mu)

    assert panels[2].series[0].values == pytest.approx([radius, radius])
    assert panels[3].series[0].values == pytest.approx([radius, radius])
    assert panels[3].series[1].values == pytest.approx([0.0, 0.0], abs=1.0e-12)


def test_diagnostics_figure_uses_shared_time_subplots() -> None:
    trajectory = np.asarray(
        [
            [100.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [50.0, 25.0, 0.0, 0.0, 0.1, 0.0, 10.0],
        ]
    )

    figure = trajectory_diagnostics_figure(
        trajectory,
        frame_kind="relative",
    )

    assert len(figure.data) == 8
    assert figure.layout.xaxis3.title.text == "Time (s)"
