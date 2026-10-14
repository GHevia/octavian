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
