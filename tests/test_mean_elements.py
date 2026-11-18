"""Mean-element transforms checked against independent Cartesian integration."""

from dataclasses import replace

import numpy as np
import pytest

from octavian import (
    EARTH,
    MeanElements,
    Perturbations,
    mean_to_osculating,
    osculating_to_mean,
    propagate,
    state,
)
from octavian.astro.equinoctial import equinoctial_to_cartesian


def _elements(e=0.05, inc=40.0):
    return MeanElements(
        7_000_000.0, e * np.cos(0.4), e * np.sin(0.4), np.tan(np.deg2rad(inc / 2)), 0.0, 1.2
    )


@pytest.mark.parametrize("e,inc", [(0, 0), (0, 90), (0.1, 30), (0.6, 150)])
def test_mean_osculating_round_trip(e, inc):
    # Keep even the eccentric test orbit above Earth's surface.
    initial = replace(_elements(e, inc), p_m=12_000_000.0)
    cartesian = mean_to_osculating(initial, samples=256)
    recovered = osculating_to_mean(cartesian, samples=256)
    np.testing.assert_allclose(recovered.as_array(), initial.as_array(), rtol=1e-11, atol=2e-12)
    back = mean_to_osculating(recovered, samples=256)
    np.testing.assert_allclose(back.r_m, cartesian.r_m, atol=1e-5)
    np.testing.assert_allclose(back.v_mps, cartesian.v_mps, atol=1e-8)


def test_zero_j2_is_keplerian_with_unwrapped_longitude():
    initial = _elements()
    n = np.sqrt(EARTH.mu_m3ps2 / initial.a_m**3)
    times = np.linspace(0, 4 * np.pi / n, 21)
    mean = propagate.mean_elements(initial, times, j2=False)
    np.testing.assert_allclose(mean.elements[:, :5], np.tile(initial.as_array()[:5], (21, 1)))
    assert mean.elements[-1, 5] == pytest.approx(initial.mean_longitude_rad + 4 * np.pi)
    cartesian = mean_to_osculating(initial, j2=False)
    numerical = propagate.two_body(cartesian, times, mu_m3ps2=EARTH.mu_m3ps2)
    np.testing.assert_allclose(mean.to_osculating(), numerical, atol=2e-6)
    np.testing.assert_allclose(
        osculating_to_mean(cartesian, j2=False).as_array(), initial.as_array(), atol=1e-8
    )


def test_secular_rates_and_backward_propagation():
    initial = _elements()
    times = np.array([0.0, 10_000.0])
    result = propagate.mean_elements(initial, times)
    values = result.elements[-1]
    n = np.sqrt(EARTH.mu_m3ps2 / initial.a_m**3)
    factor = 0.75 * EARTH.j2_coefficient * n * (EARTH.mean_radius_m / initial.p_m) ** 2
    cosine = np.cos(np.deg2rad(40))
    node_rate = -2 * factor * cosine
    periapsis_rate = factor * (5 * cosine**2 - 1)
    assert np.arctan2(values[4], values[3]) / times[-1] == pytest.approx(node_rate)
    assert (np.arctan2(values[2], values[1]) - 0.4) / times[-1] == pytest.approx(
        node_rate + periapsis_rate
    )
    assert values[0] == initial.p_m
    assert np.hypot(values[1], values[2]) == pytest.approx(np.hypot(initial.f, initial.g))
    backward = propagate.mean_elements(MeanElements(*values), [0, -10_000])
    np.testing.assert_allclose(backward.elements[-1], initial.as_array(), atol=1e-9)


@pytest.mark.parametrize("e,inc", [(0, 0), (0.05, 40), (0.2, 90)])
def test_j2_truncation_error_is_second_order(e, inc):
    initial = _elements(e, inc)
    period = 2 * np.pi * np.sqrt(initial.a_m**3 / EARTH.mu_m3ps2)
    times = np.linspace(0, period, 31)
    errors = []
    for fraction in (1.0, 0.5):
        body = replace(EARTH, j2_coefficient=EARTH.j2_coefficient * fraction)
        cartesian = mean_to_osculating(initial, central_body=body)
        numerical = propagate.inertial(
            cartesian, times, central_body=body, perturbations=Perturbations(j2=True), max_step_s=5
        )
        reconstructed = propagate.mean_elements(initial, times, central_body=body).to_osculating()
        errors.append(np.max(np.linalg.norm(numerical[:, :3] - reconstructed[:, :3], axis=1)))
    assert errors[0] < 500.0
    assert 3.8 < errors[0] / errors[1] < 4.2


def test_transform_fourier_sampling_converges():
    initial = replace(_elements(0.6, 70), p_m=12_000_000.0)
    low, medium, high = [mean_to_osculating(initial, samples=n) for n in (128, 256, 512)]
    coarse = np.linalg.norm(low.r_m - medium.r_m)
    fine = np.linalg.norm(medium.r_m - high.r_m)
    assert fine < 1e-5
    assert fine < coarse / 100


def test_batched_equinoctial_inverse_matches_scalars():
    initial = _elements()
    angles = np.linspace(-np.pi, np.pi, 17)
    rows = equinoctial_to_cartesian(initial.as_array()[:5], angles, EARTH.mu_m3ps2)
    for angle, row in zip(angles, rows, strict=True):
        np.testing.assert_allclose(
            row, equinoctial_to_cartesian(initial.as_array()[:5], angle, EARTH.mu_m3ps2)
        )


@pytest.mark.parametrize("times", [[], [0, 0], [0, np.nan], [[0, 1]], [0, 2, 1]])
def test_rejects_invalid_times(times):
    with pytest.raises(ValueError, match="times_s"):
        propagate.mean_elements(_elements(), times)


def test_rejects_invalid_models_and_domains():
    with pytest.raises(ValueError, match="positive p"):
        replace(_elements(), p_m=-1)
    with pytest.raises(ValueError, match="e < 1"):
        replace(_elements(), f=1, g=0)
    with pytest.raises(ValueError, match="samples"):
        mean_to_osculating(_elements(), samples=31)
    with pytest.raises(ValueError, match="too large"):
        mean_to_osculating(replace(_elements(), p_m=1_000))
    with pytest.raises(ValueError, match="retrograde"):
        osculating_to_mean(state([7e6, 0, 0], [0, -7_500, 0]))
