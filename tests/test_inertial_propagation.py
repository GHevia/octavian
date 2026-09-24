"""Numerical inertial propagation and the optional force model."""

import numpy as np
import pytest

from octavian import EARTH, Cannonball, Perturbations, Spacecraft, propagate, state
from octavian.astro import classical_to_cartesian, propagate_cartesian_rv

MU = EARTH.mu_m3ps2


@pytest.mark.parametrize("direction", [1, -1])
def test_circular_equatorial_orbit_matches_analytic_history(direction):
    radius = 7_000_000.0
    motion = np.sqrt(MU / radius**3)
    times = direction * np.linspace(0, 2 * np.pi / motion, 121)
    initial = state([radius, 0, 0], [0, radius * motion, 0])
    history = propagate.inertial(initial, times)
    expected_positions = radius * np.column_stack(
        [
            np.cos(motion * times),
            np.sin(motion * times),
            np.zeros_like(times),
        ]
    )
    np.testing.assert_allclose(history[:, :3], expected_positions, atol=0.03, rtol=0)
    np.testing.assert_array_equal(history[:, 6], times)
    reversed_history = propagate.inertial(initial, times[::-1])
    np.testing.assert_allclose(reversed_history, history[::-1], atol=1e-9)


def test_inclined_eccentric_orbit_agrees_with_kepler_propagation():
    position, velocity = classical_to_cartesian(
        a_m=9e6,
        e=0.2,
        inc_deg=35,
        raan_deg=40,
        argp_deg=20,
        true_anomaly_deg=70,
        mu_m3ps2=MU,
    )
    initial = state(position, velocity)
    times = np.linspace(0, 6000, 101)
    actual = propagate.inertial(initial, times)
    expected = np.asarray(
        [propagate_cartesian_rv(np.hstack([position, velocity]), time, MU) for time in times]
    )
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=0.1, rtol=0)
    np.testing.assert_allclose(actual[:, 3:6], expected[:, 3:6], atol=0.0001, rtol=0)


@pytest.mark.parametrize(
    "perturbations",
    [
        Perturbations(j2=True),
        Perturbations(moon=True, sun=True),
        Perturbations(drag=True),
        Perturbations(srp=True),
    ],
)
def test_inertial_perturbations_match_existing_absolute_force_model(perturbations):
    radius = EARTH.mean_radius_m + 300_000
    initial = state([radius, 0, 0], [0, np.sqrt(MU / radius), 100])
    vehicle = Spacecraft(
        dry_mass_kg=100,
        cannonball=Cannonball(drag_area_m2=10, srp_area_m2=10),
    )
    times = np.linspace(0, 300, 31)
    kwargs = dict(perturbations=perturbations, initial_epoch="2026-01-01T00:00:00Z")
    history = propagate.inertial(initial, times, spacecraft=vehicle, **kwargs)
    paired = propagate.relative(
        initial,
        None,
        times,
        deputy_initial_eci=initial,
        chief_spacecraft=vehicle,
        deputy_spacecraft=vehicle,
        **kwargs,
    )
    np.testing.assert_allclose(history, paired.deputy_trajectory_eci, atol=1e-7, rtol=0)
    unperturbed = propagate.inertial(initial, times)
    assert np.linalg.norm(history[-1, :3] - unperturbed[-1, :3]) > 1e-4


@pytest.mark.parametrize(
    "times,kwargs,message",
    [
        ([1, 2], {}, "first or last"),
        ([0, 1, 1], {}, "strictly monotonic"),
        ([0, np.nan], {}, "finite"),
        ([0, 1], {"max_step_s": 0}, "max_step_s"),
        ([0, 1], {"ephemeris_step_s": np.inf}, "ephemeris_step_s"),
        ([0, 1], {"perturbations": Perturbations(moon=True)}, "initial_epoch"),
        ([0, 1], {"perturbations": Perturbations(drag=True)}, "spacecraft"),
    ],
)
def test_invalid_inertial_configuration(times, kwargs, message):
    with pytest.raises(ValueError, match=message):
        propagate.inertial(state([7e6, 0, 0], [0, 7500, 0]), times, **kwargs)
