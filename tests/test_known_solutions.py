from __future__ import annotations

import math

import numpy as np
import pytest

from octavian.data.ephemeris import sample_sun_moon_positions_eci_tod
from octavian.dynamics import (
    MOON_MU_M3PS2,
    SUN_MU_M3PS2,
    j2_acceleration_components,
    third_body_acceleration_components,
)

MU = 3.986004418e14
R_EARTH_M = 6_378_136.3
J2 = 1.08262668e-3


def hohmann_transfer_reference(
    *,
    r_initial_m: float,
    r_final_m: float,
    mu_m3ps2: float = MU,
) -> tuple[float, float, float, float]:
    """Return dv1, dv2, total dv, and half-period TOF for a circular Hohmann transfer."""
    transfer_a_m = 0.5 * (r_initial_m + r_final_m)
    circular_initial_mps = math.sqrt(mu_m3ps2 / r_initial_m)
    circular_final_mps = math.sqrt(mu_m3ps2 / r_final_m)
    periapsis_transfer_mps = math.sqrt(mu_m3ps2 * (2.0 / r_initial_m - 1.0 / transfer_a_m))
    apoapsis_transfer_mps = math.sqrt(mu_m3ps2 * (2.0 / r_final_m - 1.0 / transfer_a_m))
    dv1_mps = periapsis_transfer_mps - circular_initial_mps
    dv2_mps = circular_final_mps - apoapsis_transfer_mps
    tof_s = math.pi * math.sqrt((transfer_a_m**3) / mu_m3ps2)
    return dv1_mps, dv2_mps, dv1_mps + dv2_mps, tof_s


def test_hohmann_reference_for_7000_to_12000_km_circular_orbits() -> None:
    dv1_mps, dv2_mps, total_dv_mps, tof_s = hohmann_transfer_reference(
        r_initial_m=7_000e3,
        r_final_m=12_000e3,
    )

    assert dv1_mps == pytest.approx(934.978, abs=1.0e-3)
    assert dv2_mps == pytest.approx(816.125, abs=1.0e-3)
    assert total_dv_mps == pytest.approx(1_751.103, abs=1.0e-3)
    assert tof_s == pytest.approx(4_607.511, abs=1.0e-3)


def test_j2_acceleration_matches_closed_form_equator_and_pole() -> None:
    equator_accel = np.array(
        j2_acceleration_components(
            [7_000e3, 0.0, 0.0],
            mu_m3ps2=MU,
            radius_m=R_EARTH_M,
            j2=J2,
        )
    )
    pole_accel = np.array(
        j2_acceleration_components(
            [0.0, 0.0, 7_000e3],
            mu_m3ps2=MU,
            radius_m=R_EARTH_M,
            j2=J2,
        )
    )

    expected_magnitude = 1.5 * J2 * MU * (R_EARTH_M**2) / (7_000e3**4)
    np.testing.assert_allclose(equator_accel, [-expected_magnitude, 0.0, 0.0], rtol=1.0e-14)
    np.testing.assert_allclose(pole_accel, [0.0, 0.0, 2.0 * expected_magnitude], rtol=1.0e-14)


def test_core_perturbation_accelerations_have_expected_magnitudes() -> None:
    spacecraft_position_m = np.array([7_000e3, 0.0, 0.0])
    _, positions_m = sample_sun_moon_positions_eci_tod(
        initial_epoch="2026-01-01T00:00:00Z",
        duration_s=3600.0,
        step_s=3600.0,
    )
    sun_position_m = positions_m["sun"][0]
    moon_position_m = positions_m["moon"][0]

    j2_accel = np.array(
        j2_acceleration_components(
            spacecraft_position_m,
            mu_m3ps2=MU,
            radius_m=R_EARTH_M,
            j2=J2,
        )
    )
    sun_accel = np.array(
        third_body_acceleration_components(
            spacecraft_position_m,
            sun_position_m,
            mu_m3ps2=SUN_MU_M3PS2,
        )
    )
    moon_accel = np.array(
        third_body_acceleration_components(
            spacecraft_position_m,
            moon_position_m,
            mu_m3ps2=MOON_MU_M3PS2,
        )
    )

    j2_magnitude = float(np.linalg.norm(j2_accel))
    sun_magnitude = float(np.linalg.norm(sun_accel))
    moon_magnitude = float(np.linalg.norm(moon_accel))

    sun_tidal_scale = SUN_MU_M3PS2 * np.linalg.norm(spacecraft_position_m) / (
        np.linalg.norm(sun_position_m) ** 3
    )
    moon_tidal_scale = MOON_MU_M3PS2 * np.linalg.norm(spacecraft_position_m) / (
        np.linalg.norm(moon_position_m) ** 3
    )

    assert j2_magnitude == pytest.approx(1.0967e-2, rel=2.0e-3)
    assert sun_magnitude == pytest.approx(sun_tidal_scale, rel=2.0)
    assert moon_magnitude == pytest.approx(moon_tidal_scale, rel=2.0)
    assert 1.0e-8 < sun_magnitude < 1.0e-6
    assert 1.0e-8 < moon_magnitude < 1.0e-5
    assert j2_magnitude > 1.0e4 * max(sun_magnitude, moon_magnitude)
