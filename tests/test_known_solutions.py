from __future__ import annotations

import math

import numpy as np
import pytest

from octavian.dynamics import j2_acceleration_components

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
