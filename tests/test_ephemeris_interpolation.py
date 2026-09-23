from __future__ import annotations

import numpy as np
import pytest

from octavian.dynamics import ThirdBodyTable
from octavian.relative import SolarDirectionTable


@pytest.fixture(params=["third_body", "solar_position", "solar_direction"])
def interpolate(request):
    times = np.array([0.0, 10.0])
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if request.param == "third_body":
        return ThirdBodyTable("sun", 1.0, None, times, vectors).position_at
    solar = SolarDirectionTable(times, vectors, vectors)
    return solar.sun_position_at if request.param == "solar_position" else solar.at


def test_ephemeris_endpoints_accept_solver_roundoff(interpolate) -> None:
    assert interpolate(-1.0e-10) == pytest.approx([1.0, 0.0, 0.0])
    assert interpolate(10.0 + 1.0e-10) == pytest.approx([0.0, 1.0, 0.0])
    assert np.all(np.isfinite(interpolate(5.0)))


@pytest.mark.parametrize("time_s", [-1.0e-4, 10.0001])
def test_ephemeris_still_rejects_extrapolation(interpolate, time_s: float) -> None:
    with pytest.raises(ValueError, match="outside the sampled time range"):
        interpolate(time_s)


@pytest.mark.parametrize("time_s", [float("nan"), float("inf")])
def test_ephemeris_rejects_non_finite_times(interpolate, time_s: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        interpolate(time_s)


def test_solar_roundoff_clipping_preserves_query_shape() -> None:
    table = SolarDirectionTable(
        np.array([0.0, 10.0]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    result = table.at(np.array([[-1.0e-10, 10.0 + 1.0e-10]]))
    assert result.shape == (1, 2, 3)
    assert result[0] == pytest.approx(table.directions_ric)
