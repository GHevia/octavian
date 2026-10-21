from __future__ import annotations

import numpy as np
import pytest

from octavian import guesses
from octavian.config.declarations import build_initial_guess
from octavian.solvers.compiler.qlaw_guessing import (
    build_low_thrust_qlaw_seed,
    equinoctial_elements,
    gauss_matrix,
)

MU = 3.986004418e14


@pytest.mark.parametrize(
    "kwargs",
    [
        {"throttle": 0},
        {"throttle": float("nan")},
        {"steps_per_orbit": 19},
        {"orbit_samples": 2},
        {"plane_tol_deg": -1},
        {"eccentricity_tol": float("inf")},
    ],
)
def test_qlaw_declaration_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        guesses.low_thrust_qlaw(**kwargs)


def test_qlaw_config_declaration():
    config = build_initial_guess({"type": "low_thrust_qlaw", "plane_tol_deg": 0.2}, "guess")
    assert config == guesses.low_thrust_qlaw(plane_tol_deg=0.2)


def test_equinoctial_gauss_matrix_matches_cartesian_velocity_derivatives():
    # Eccentric, inclined state exercises every element and thrust direction.
    rv = np.array([7e6, 1e6, 2e6, -1200, 7200, 1500.0])
    elements, longitude = equinoctial_elements(rv, MU)
    radial = rv[:3] / np.linalg.norm(rv[:3])
    normal = np.cross(rv[:3], rv[3:6])
    normal /= np.linalg.norm(normal)
    basis = np.column_stack([radial, np.cross(normal, radial), normal])
    numerical = []
    for direction in basis.T:
        plus, minus = rv.copy(), rv.copy()
        plus[3:6] += direction * 0.01
        minus[3:6] -= direction * 0.01
        numerical.append(
            (equinoctial_elements(plus, MU)[0] - equinoctial_elements(minus, MU)[0]) / 0.02
        )
    scale = np.array([elements[0], 1, 1, 1, 1])[:, None]
    np.testing.assert_allclose(
        gauss_matrix(elements, np.asarray(longitude), MU) / scale,
        np.asarray(numerical).T / scale,
        rtol=2e-7,
        atol=1e-12,
    )


def test_qlaw_seed_raises_circular_equatorial_orbit_and_changes_plane():
    initial = np.array([7e6, 0, 0, 0, np.sqrt(MU / 7e6), 0])
    angle = np.deg2rad(3)
    target = np.array(
        [8e6, 0, 0, 0, np.sqrt(MU / 8e6) * np.cos(angle), np.sqrt(MU / 8e6) * np.sin(angle)]
    )
    rows, info = build_low_thrust_qlaw_seed(
        initial_rv=initial,
        target_rv=target,
        mu_m3ps2=MU,
        initial_mass_kg=560,
        dry_mass_kg=500,
        thrust_N=5,
        isp_s=1800,
        time_bounds_s=(3600, 2 * 86400),
        npts=301,
        config=guesses.low_thrust_qlaw(plane_tol_deg=0.05),
    )
    rows = np.asarray(rows)
    assert info["seed_target_reached"]
    assert info["seed_plane_error_deg"] <= 0.05
    assert info["seed_semi_major_axis_relative_error"] < 0.001
    assert info["seed_eccentricity_vector_error"] < 0.005
    assert rows.shape == (301, 11)
    assert np.all(np.diff(rows[:, 7]) > 0)
    assert np.all(np.diff(rows[:, 6]) <= 0)
    assert rows[-1, 6] > 500
    # Includes cross-track firing, not merely a rotated plotting geometry.
    assert np.max(np.abs(rows[:, 10])) > 0.1
    throttle = np.linalg.norm(rows[:, 8:11], axis=1)
    assert np.max(throttle) <= 0.8 + 1e-10
    np.testing.assert_allclose(
        rows[-1, 6], 560 - 5 * 0.8 * rows[-1, 7] / (1800 * 9.80665), atol=1e-7
    )


def test_qlaw_rejects_unbound_and_retrograde_orbits():
    for rv in [
        np.array([7e6, 0, 0, 0, -np.sqrt(MU / 7e6), 0]),
        np.array([7e6, 0, 0, 0, 2 * np.sqrt(MU / 7e6), 0]),
    ]:
        with pytest.raises(ValueError):
            equinoctial_elements(rv, MU)


def test_qlaw_coasts_to_minimum_duration_and_returns_partial_seeds():
    initial = np.array([7e6, 0, 0, 0, np.sqrt(MU / 7e6), 0])
    common = dict(
        initial_rv=initial,
        mu_m3ps2=MU,
        initial_mass_kg=560,
        dry_mass_kg=500,
        thrust_N=5,
        isp_s=1800,
        npts=51,
        config=guesses.low_thrust_qlaw(),
    )
    rows, info = build_low_thrust_qlaw_seed(target_rv=initial, time_bounds_s=(3600, 7200), **common)
    assert info["seed_target_reached"] and info["seed_tof_s"] == 3600
    np.testing.assert_allclose(np.asarray(rows)[:, 6], 560)
    np.testing.assert_allclose(np.asarray(rows)[:, 8:11], 0)
    target = np.array([8e6, 0, 0, 0, np.sqrt(MU / 8e6), 0])
    rows, info = build_low_thrust_qlaw_seed(target_rv=target, time_bounds_s=(1, 100), **common)
    assert not info["seed_target_reached"] and info["seed_tof_s"] == 100
    common["initial_mass_kg"] = 500.001
    with pytest.raises(ValueError, match="propellant"):
        build_low_thrust_qlaw_seed(target_rv=target, time_bounds_s=(1, 100), **common)


def test_qlaw_compiler_preserves_steering_in_ric_frame():
    from octavian import (
        Dynamics,
        LowThrustQLawGuess,
        Phase,
        Spacecraft,
        ThrustControl,
        Thruster,
        state,
    )
    from octavian.relative.transforms import ric_basis
    from octavian.solvers.composable import _build_guess_single_phase_low_thrust

    initial = state([7e6, 0, 0], [0, np.sqrt(MU / 7e6), 0])
    angle = np.deg2rad(5)
    target = state(
        [8e6, 0, 0], [0, np.sqrt(MU / 8e6) * np.cos(angle), np.sqrt(MU / 8e6) * np.sin(angle)]
    )
    phase = Phase(
        name="plane_change",
        mode="low_thrust",
        dynamics=Dynamics(mu_m3ps2=MU),
        spacecraft=Spacecraft(
            dry_mass_kg=500, thrusters=[Thruster(thrust_N=5, isp_s=1800, propellant_mass_kg=60)]
        ),
        initial_state=initial,
        final_state=target,
        initial_guess=guesses.low_thrust_qlaw(),
    )
    assert isinstance(phase.initial_guess, LowThrustQLawGuess)
    inertial, info = _build_guess_single_phase_low_thrust(
        phase, mu=MU, tf_bounds=(1, 100), nsegs=10
    )
    phase.thrust_control = ThrustControl.vector(frame="ric")
    local, _ = _build_guess_single_phase_low_thrust(phase, mu=MU, tf_bounds=(1, 100), nsegs=10)
    for eci, ric in zip(inertial, local, strict=True):
        np.testing.assert_allclose(ric[8:11], ric_basis(eci[:3], eci[3:6]) @ eci[8:11], atol=1e-12)
    import json

    json.dumps(info)  # Partial-seed metadata must remain portable.


def test_plane_change_steering_reverses_normal_thrust_at_opposite_nodes():
    from octavian.solvers.compiler.qlaw_guessing import qlaw_direction

    initial = np.array([7e6, 0, 0, 0, np.sqrt(MU / 7e6), 0])
    target = initial.copy()
    target[4:6] = np.sqrt(MU / 7e6) * np.array([np.cos(0.1), np.sin(0.1)])
    elements, _ = equinoctial_elements(target, MU)
    assert qlaw_direction(initial, elements, MU, 32)[2] > 0.9
    assert qlaw_direction(-initial, elements, MU, 32)[2] < -0.9


@pytest.mark.parametrize(
    "elements,target",
    [
        ([7.3e6, 0.035, 0.021, 0.12, -0.07], [8e6, 0.01, -0.015, 0.08, 0.02]),
        ([7.3e6, 0.2, -0.1, 1.7, -0.5], [8e6, -0.03, 0.1, -0.4, 0.2]),
    ],
)
def test_analytic_quotient_gradient_matches_independent_finite_differences(elements, target):
    from octavian.solvers.compiler.qlaw_guessing import proximity_quotient_gradient

    elements, target = np.array(elements), np.array(target)
    _, gradient = proximity_quotient_gradient(elements, target, MU, 64)
    for i, step in enumerate([10.0, 1e-6, 1e-6, 1e-6, 1e-6]):
        plus, minus = elements.copy(), elements.copy()
        plus[i] += step
        minus[i] -= step
        numerical = (
            proximity_quotient_gradient(plus, target, MU, 64)[0]
            - proximity_quotient_gradient(minus, target, MU, 64)[0]
        ) / (2 * step)
        assert gradient[i] == pytest.approx(numerical, rel=1e-6)


def test_quotient_cannot_reduce_plane_error_cost_by_inflating_the_orbit():
    from octavian.solvers.compiler.qlaw_guessing import proximity_quotient_gradient

    target = np.array([7e6, 0, 0, np.tan(np.deg2rad(40) / 2), 0])
    values = []
    for scale in (1, 2, 5, 10, 100):
        current = np.array([7e6 * scale, 0, 0, 0, 0])
        value, gradient = proximity_quotient_gradient(current, target, MU, 64)
        values.append(value)
        assert gradient[0] >= 0
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("inclination", [40, 120, 170])
def test_plane_only_steering_does_not_raise_the_orbit_or_turn_toward_retrograde(inclination):
    from octavian.solvers.compiler.qlaw_guessing import (
        equinoctial_to_cartesian,
        qlaw_direction,
    )

    equatorial = np.array([7e6, 0, 0, 0, 0])
    inclined = np.array([7e6, 0, 0, np.tan(np.deg2rad(inclination) / 2), 0])
    for current, target, sign in ((equatorial, inclined, 1), (inclined, equatorial, -1)):
        direction = qlaw_direction(equinoctial_to_cartesian(current, 0, MU), target, MU, 64)
        np.testing.assert_allclose(direction, [0, 0, sign], atol=1e-10)


@pytest.mark.parametrize("averaged", [False, True])
@pytest.mark.parametrize(
    "inc0,inc1,node0,node1,a0,a1,e0,e1",
    [
        (0, 40, 0, 0, 40e6, 42e6, 0, 0),
        (0, 60, 0, 0, 40e6, 42e6, 0, 0),
        (0, 120, 0, 0, 40e6, 42e6, 0, 0),
        (120, 0, 0, 0, 42e6, 40e6, 0, 0),
        (0, 170, 0, 0, 40e6, 40e6, 0, 0),
        (0, 180, 0, 0, 40e6, 40e6, 0, 0),
        (180, 0, 0, 0, 40e6, 40e6, 0, 0),
        (80, 80, 0, 120, 40e6, 42e6, 0, 0),
        # The short plane rotation passes through the inertial chart's pole.
        (170, 170, 0, 180, 40e6, 40e6, 0, 0),
        # Exactly antipodal normals require choosing a nonzero descent direction.
        (90, 90, 0, 180, 40e6, 40e6, 0, 0),
        (30, 90, 10, 80, 42e6, 40e6, 0.1, 0.2),
    ],
)
def test_large_plane_change_seeds_reach_the_physical_target(
    averaged, inc0, inc1, node0, node1, a0, a1, e0, e1
):
    from octavian.astro import classical_to_cartesian

    def cartesian(a, e, inc, node):
        return np.concatenate(
            classical_to_cartesian(
                a_m=a,
                e=e,
                inc_deg=inc,
                raan_deg=node,
                argp_deg=17,
                true_anomaly_deg=11,
                mu_m3ps2=MU,
            )
        )

    initial, target = cartesian(a0, e0, inc0, node0), cartesian(a1, e1, inc1, node1)
    rows, info = build_low_thrust_qlaw_seed(
        initial_rv=initial,
        target_rv=target,
        mu_m3ps2=MU,
        initial_mass_kg=23,
        dry_mass_kg=5,
        thrust_N=0.1,
        isp_s=3000,
        time_bounds_s=(0, 75 * 86400),
        npts=1601,
        config=guesses.low_thrust_qlaw(
            averaged=averaged, throttle=0.8, plane_tol_deg=0.1, orbit_samples=64
        ),
    )
    rows = np.asarray(rows)
    assert info["seed_target_reached"]
    assert info["seed_tof_s"] < 75 * 86400
    assert 1600 / info["seed_estimated_revolutions"] >= 16
    assert np.all(np.isfinite(rows))
    np.testing.assert_allclose(rows[0, :6], initial, rtol=1e-12, atol=1e-7)
    assert np.all(np.diff(rows[:, 7]) > 0)
    assert np.all(np.diff(rows[:, 6]) <= 1e-12)
    assert rows[-1, 6] > 5
    assert np.max(np.linalg.norm(rows[:, 8:11], axis=1)) <= 0.8 + 1e-12
    # Check the returned inertial state independently of the seed's metadata.
    r, v = rows[-1, :3], rows[-1, 3:6]
    momentum = np.cross(r, v)
    target_momentum = np.cross(target[:3], target[3:6])
    cosine = momentum @ target_momentum / np.linalg.norm(momentum) / np.linalg.norm(target_momentum)
    assert np.rad2deg(np.arccos(np.clip(cosine, -1, 1))) <= 0.1 + 1e-8
    final_a = 1 / (2 / np.linalg.norm(r) - v @ v / MU)
    assert abs(final_a / a1 - 1) <= 0.001
    eccentricity = np.cross(v, momentum) / MU - r / np.linalg.norm(r)
    target_eccentricity = np.cross(target[3:6], target_momentum) / MU - target[:3] / np.linalg.norm(
        target[:3]
    )
    assert np.linalg.norm(eccentricity - target_eccentricity) <= 0.005


@pytest.mark.parametrize("eccentricity", [0, 0.1, 0.7])
def test_equinoctial_and_mean_longitude_round_trip(eccentricity):
    from octavian.solvers.compiler.qlaw_guessing import (
        equinoctial_to_cartesian,
        mean_longitude,
        true_longitude,
    )

    elements = np.array([7e6, eccentricity * np.cos(0.4), eccentricity * np.sin(0.4), 0.3, -0.1])
    for longitude in np.linspace(-np.pi, 4 * np.pi, 20):
        rv = equinoctial_to_cartesian(elements, longitude, MU)
        recovered, angle = equinoctial_elements(rv, MU)
        np.testing.assert_allclose(recovered, elements, rtol=1e-12, atol=1e-8)
        mean = mean_longitude(elements, longitude)
        rebuilt = equinoctial_to_cartesian(elements, true_longitude(elements, mean), MU)
        np.testing.assert_allclose(rebuilt, rv, rtol=1e-12, atol=1e-7)


@pytest.mark.parametrize(
    "inclination0,inclination1,radius0,radius1",
    [
        (0, 2, 7e6, 7.2e6),
        (35, 33, 7.2e6, 7e6),
        (65, 67, 7e6, 7.2e6),
    ],
)
def test_averaged_qlaw_handles_raising_lowering_and_inclined_starts(
    inclination0,
    inclination1,
    radius0,
    radius1,
):
    def circular(radius, inclination):
        angle = np.deg2rad(inclination)
        return np.array(
            [
                radius,
                0,
                0,
                0,
                np.sqrt(MU / radius) * np.cos(angle),
                np.sqrt(MU / radius) * np.sin(angle),
            ]
        )

    rows, info = build_low_thrust_qlaw_seed(
        initial_rv=circular(radius0, inclination0),
        target_rv=circular(radius1, inclination1),
        mu_m3ps2=MU,
        initial_mass_kg=23,
        dry_mass_kg=20,
        thrust_N=0.1,
        isp_s=1800,
        time_bounds_s=(3600, 5 * 86400),
        npts=301,
        config=guesses.low_thrust_qlaw(averaged=True, plane_tol_deg=0.05, orbit_samples=64),
    )
    assert info["seed_target_reached"]
    assert info["seed_integration_steps"] < 1000
    assert info["seed_averaged"]
    assert np.max(np.linalg.norm(np.asarray(rows)[:, 8:11], axis=1)) <= 0.8 + 1e-12


def test_averaged_qlaw_scales_to_hundreds_of_revolutions_at_100_millinewtons():
    angle = np.deg2rad(2)
    rows, info = build_low_thrust_qlaw_seed(
        initial_rv=np.array([7e6, 0, 0, 0, np.sqrt(MU / 7e6), 0]),
        target_rv=np.array(
            [
                7.2e6,
                0,
                0,
                0,
                np.sqrt(MU / 7.2e6) * np.cos(angle),
                np.sqrt(MU / 7.2e6) * np.sin(angle),
            ]
        ),
        mu_m3ps2=MU,
        initial_mass_kg=560,
        dry_mass_kg=500,
        thrust_N=0.1,
        isp_s=1800,
        time_bounds_s=(3600, 50 * 86400),
        npts=2001,
        config=guesses.low_thrust_qlaw(averaged=True, plane_tol_deg=0.05, orbit_samples=64),
    )
    assert info["seed_target_reached"]
    assert info["seed_estimated_revolutions"] > 400
    assert info["seed_integration_steps"] < 3000
    assert info["seed_final_mass_kg"] > 500
    assert np.all(np.diff(np.asarray(rows)[:, 7]) > 0)


@pytest.mark.parametrize("averaged", [False, True])
def test_long_coasts_preserve_two_body_orbit_without_secular_energy_drift(averaged):
    initial = np.array([7e6, 0, 0, 0, np.sqrt(MU / 7e6), 0])
    period = 2 * np.pi * np.sqrt(7e6**3 / MU)
    rows, info = build_low_thrust_qlaw_seed(
        initial_rv=initial,
        target_rv=initial,
        mu_m3ps2=MU,
        initial_mass_kg=23,
        dry_mass_kg=20,
        thrust_N=0.1,
        isp_s=1800,
        time_bounds_s=(20 * period, 21 * period),
        npts=301,
        config=guesses.low_thrust_qlaw(averaged=averaged),
    )
    assert info["seed_target_reached"]
    rows = np.asarray(rows)
    np.testing.assert_allclose(rows[:, 6], 23)
    np.testing.assert_allclose(rows[:, 8:11], 0)
    assert info["seed_eccentricity_vector_error"] < 1e-12
    assert info["seed_semi_major_axis_relative_error"] < 1e-12
    assert np.max(abs(np.linalg.norm(rows[:, :3], axis=1) - 7e6)) < 1
