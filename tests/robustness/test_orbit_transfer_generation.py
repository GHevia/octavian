from __future__ import annotations

import numpy as np
import pytest

from .orbit_transfers import (
    ABOVE_GEO_MIN_PERIGEE_M,
    DEFAULT_CAMPAIGN_SEED,
    EARTH_RADIUS_M,
    MIN_PERIGEE_ALTITUDE_M,
    generate_transfer_scenarios,
)


def test_campaign_generation_is_deterministic() -> None:
    first = generate_transfer_scenarios(12, seed=DEFAULT_CAMPAIGN_SEED)
    second = generate_transfer_scenarios(12, seed=DEFAULT_CAMPAIGN_SEED)

    assert [case.to_dict() for case in first] == [case.to_dict() for case in second]
    assert len({case.case_seed for case in first}) == len(first)


def test_generated_orbits_are_bound_and_clear_the_earth() -> None:
    minimum_perigee_m = EARTH_RADIUS_M + MIN_PERIGEE_ALTITUDE_M
    scenarios = generate_transfer_scenarios(100)

    for scenario in scenarios:
        for orbit in (scenario.initial_orbit, scenario.final_orbit):
            assert 0.0 <= orbit.e < 1.0
            assert orbit.perigee_radius_m >= minimum_perigee_m
            assert 0.0 <= orbit.inc_deg <= 70.0

        initial_state, final_state = scenario.boundary_states()
        for vector in (
            initial_state.r_m,
            initial_state.v_mps,
            final_state.r_m,
            final_state.v_mps,
        ):
            assert np.all(np.isfinite(vector))


def test_hundred_case_campaign_covers_public_transfer_knobs() -> None:
    scenarios = generate_transfer_scenarios(100)

    assert {case.backend for case in scenarios} == {"quick", "composable"}
    assert {case.link_kind for case in scenarios} == {"direct", "continuous", "impulsive"}
    assert {case.nsegs for case in scenarios} == {40, 60, 80}
    assert {case.lambert_grid_size for case in scenarios} == {40, 60, 80, 100}
    assert {case.nrevs_to_try for case in scenarios} == {(0,), (0, 1)}
    assert {case.tof_is_relative for case in scenarios} == {False, True}
    assert {case.time_weight for case in scenarios} == {0.0, 0.05}


def test_hundred_case_campaign_includes_above_geo_transfers() -> None:
    scenarios = generate_transfer_scenarios(100)
    high_orbit_cases = [case for case in scenarios if case.orbit_regime == "above_geo"]

    assert len(high_orbit_cases) == 20
    assert {case.backend for case in high_orbit_cases} == {"quick", "composable"}
    assert {case.link_kind for case in high_orbit_cases} == {
        "direct",
        "continuous",
        "impulsive",
    }
    assert all(
        max(
            case.initial_orbit.perigee_radius_m,
            case.final_orbit.perigee_radius_m,
        )
        >= ABOVE_GEO_MIN_PERIGEE_M
        for case in high_orbit_cases
    )
    assert any(
        case.initial_orbit.perigee_radius_m >= ABOVE_GEO_MIN_PERIGEE_M
        for case in high_orbit_cases
    )
    assert any(
        case.final_orbit.perigee_radius_m >= ABOVE_GEO_MIN_PERIGEE_M
        for case in high_orbit_cases
    )


def test_campaign_rejects_non_positive_case_count() -> None:
    with pytest.raises(ValueError, match="at least one"):
        generate_transfer_scenarios(0)
