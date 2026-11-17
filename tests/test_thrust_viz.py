from __future__ import annotations

import numpy as np
import pytest

from octavian.viz.plotly import relative_trajectory_figure, save_trajectory_html

pytest.importorskip("plotly")

TRAJ = np.array([[float(t), 0, 0, 1, 0, 0, float(t)] for t in range(5)])
HISTORY = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 5], [3, 0, 0], [4, 0, 0]])


def test_throttle_coloring_has_fixed_scale_and_physical_hover_data():
    figure = relative_trajectory_figure(TRAJ, thrust_histories=[HISTORY], color_by="throttle")
    colored = [trace for trace in figure.data if trace.line.showscale]
    assert len(colored) == 1
    assert colored[0].line.cmin == 0 and colored[0].line.cmax == 1
    np.testing.assert_allclose(colored[0].line.color, HISTORY[:, 1])
    np.testing.assert_allclose(colored[0].customdata, HISTORY)


def test_firing_coloring_splits_at_threshold_without_bridging_coast():
    figure = relative_trajectory_figure(
        TRAJ, thrust_histories=[HISTORY], color_by="firing", firing_threshold=0.5
    )
    firing = next(trace for trace in figure.data if trace.name == "Firing")
    coast = next(trace for trace in figure.data if trace.name == "Coasting")
    firing_x = [x for x in firing.x if x is not None]
    assert min(firing_x) == 1.5 and max(firing_x) == 2.5
    assert None in firing.x and None in coast.x


def test_phase_boundaries_do_not_interpolate_throttle_between_burn_and_coast():
    histories = [np.array([[0, 1, 5], [2, 1, 5]]), np.array([[2, 0, 0], [4, 0, 0]])]
    figure = relative_trajectory_figure(TRAJ, thrust_histories=histories, color_by="firing")
    firing = next(trace for trace in figure.data if trace.name == "Firing")
    assert max(x for x in firing.x if x is not None) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color_by": "bad"},
        {"color_by": "throttle"},
        {"color_by": "firing", "thrust_histories": [HISTORY], "firing_threshold": -1},
        {"color_by": "throttle", "thrust_histories": [HISTORY[::-1]]},
    ],
)
def test_thrust_coloring_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        relative_trajectory_figure(TRAJ, **kwargs)


def test_inertial_html_contains_throttle_coloring(tmp_path):
    output = tmp_path / "throttle.html"
    save_trajectory_html(
        TRAJ, str(output), use_earth_texture=False, thrust_histories=[HISTORY], color_by="throttle"
    )
    html = output.read_text()
    assert "Throttle" in html and "Thrust" in html and '"showscale":true' in html
