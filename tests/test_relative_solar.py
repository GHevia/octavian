from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from octavian import EARTH, Dynamics, Mission, Phase, constraints, state
from octavian.relative import (
    SolarDirectionTable,
    circular_chief_state,
    sample_solar_directions_ric,
)
from octavian.solvers.relative_environment import build_solar_direction_tables


def _chief_state():
    radius_m = EARTH.mean_radius_m + 400_000.0
    speed_mps = np.sqrt(EARTH.mu_m3ps2 / radius_m)
    return state([radius_m, 0.0, 0.0], [0.0, speed_mps, 0.0])


def test_solar_direction_table_interpolates_unit_vectors() -> None:
    table = SolarDirectionTable(
        times_s=np.asarray([0.0, 10.0]),
        directions_ric=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    midpoint = table.at(np.asarray([5.0]))[0]

    assert np.linalg.norm(midpoint) == pytest.approx(1.0)
    assert midpoint == pytest.approx([np.sqrt(0.5), np.sqrt(0.5), 0.0])


def test_sample_solar_directions_rotates_spice_line_to_ric(monkeypatch) -> None:
    chief = _chief_state()
    mean_motion = np.sqrt(EARTH.mu_m3ps2 / np.linalg.norm(chief.r_m) ** 3)

    def fake_ephemeris(**_kwargs):
        times = np.asarray([0.0, 100.0])
        sun = np.asarray([[1.5e11, 0.0, 0.0], [1.5e11, 0.0, 0.0]])
        return times, {"sun": sun, "moon": np.zeros((2, 3))}

    monkeypatch.setattr(
        "octavian.relative.solar.sample_sun_moon_positions_eci_tod",
        fake_ephemeris,
    )
    table = sample_solar_directions_ric(
        chief_initial_state_eci=chief,
        mean_motion_radps=mean_motion,
        initial_epoch=0.0,
        duration_s=100.0,
    )

    assert table.directions_ric[0] == pytest.approx([1.0, 0.0, 0.0])
    assert not np.allclose(table.directions_ric[1], table.directions_ric[0])
    assert table.sun_position_at([50.0])[0] == pytest.approx([1.5e11, 0.0, 0.0])


def test_circular_chief_reference_preserves_radius_and_speed() -> None:
    chief = _chief_state()
    radius = np.linalg.norm(chief.r_m)
    mean_motion = np.sqrt(EARTH.mu_m3ps2 / radius**3)
    propagated = circular_chief_state(chief, 1_000.0, mean_motion)

    assert np.linalg.norm(propagated.r_m) == pytest.approx(radius)
    assert np.linalg.norm(propagated.v_mps) == pytest.approx(mean_motion * radius)
    assert np.dot(propagated.r_m, propagated.v_mps) == pytest.approx(0.0, abs=1e-5)


def test_relative_environment_requires_chief_state_for_solar_constraint() -> None:
    phase = Phase(
        dynamics=Dynamics.cwh(
            chief_orbit_radius_m=EARTH.mean_radius_m + 400_000.0,
        ),
        constraints=[constraints.solar_phase_angle(min_angle_deg=20.0)],
    )
    mission = Mission(phases=[phase], initial_epoch="2026-01-01T00:00:00Z")

    with pytest.raises(ValueError, match="chief_initial_state_eci"):
        build_solar_direction_tables(mission, [phase], [(0.0, 1_000.0)])


def test_relative_environment_samples_full_multiphase_duration(monkeypatch) -> None:
    chief = _chief_state()
    dynamics = Dynamics.cwh(
        chief_orbit_radius_m=float(np.linalg.norm(chief.r_m)),
        chief_initial_state_eci=chief,
        third_body_table_margin_s=250.0,
    )
    constrained = Phase(
        dynamics=dynamics,
        constraints=[constraints.solar_phase_angle(min_angle_deg=20.0)],
    )
    later_coast = Phase(dynamics=dynamics, previous=constrained)
    mission = Mission(
        phases=[constrained, later_coast],
        initial_epoch="2026-01-01T00:00:00Z",
    )
    sampled_durations = []

    def fake_sample_solar_directions_ric(**kwargs):
        sampled_durations.append(float(kwargs["duration_s"]))
        return SolarDirectionTable(
            times_s=np.asarray([0.0, float(kwargs["duration_s"])]),
            directions_ric=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )

    monkeypatch.setattr(
        "octavian.solvers.relative_environment.sample_solar_directions_ric",
        fake_sample_solar_directions_ric,
    )

    tables = build_solar_direction_tables(
        mission,
        [constrained, later_coast],
        [(0.0, 500.0), (500.0, 1_500.0)],
    )

    assert set(tables) == {0}
    assert sampled_durations == [1_750.0]


def test_relative_environment_uses_cumulative_relative_upper_bound(
    monkeypatch,
) -> None:
    chief = _chief_state()
    dynamics = Dynamics.relative(
        chief_initial_state_eci=chief,
        third_body_table_margin_s=250.0,
    )
    initial_coast = Phase(
        name="initial_coast",
        dynamics=dynamics,
        tof_bounds_s=(100.0, 6_000.0),
        tof_is_relative=True,
    )
    transfer = Phase(
        name="transfer",
        dynamics=dynamics,
        previous=initial_coast,
        tof_bounds_s=(1_000.0, 17_000.0),
        tof_is_relative=True,
        constraints=[constraints.solar_phase_angle(min_angle_deg=20.0)],
    )
    mission = Mission(
        phases=[initial_coast, transfer],
        initial_epoch="2026-01-01T00:00:00Z",
    )
    sampled_durations = []

    def fake_sample_solar_directions_ric(**kwargs):
        duration_s = float(kwargs["duration_s"])
        sampled_durations.append(duration_s)
        return SolarDirectionTable(
            times_s=np.asarray([0.0, duration_s]),
            directions_ric=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )

    monkeypatch.setattr(
        "octavian.solvers.relative_environment.sample_solar_directions_ric",
        fake_sample_solar_directions_ric,
    )

    tables = build_solar_direction_tables(
        mission,
        [initial_coast, transfer],
        [(100.0, 6_000.0), (1_100.0, 23_000.0)],
    )

    assert set(tables) == {1}
    assert sampled_durations == [23_250.0]


def test_cwh_rejects_perturbations_instead_of_building_a_reference() -> None:
    chief = _chief_state()
    with pytest.raises(ValueError, match=r"Dynamics\.relative"):
        Dynamics.cwh(
            chief_orbit_radius_m=float(np.linalg.norm(chief.r_m)),
            chief_initial_state_eci=chief,
            perturbations=SimpleNamespace(
                j2=True,
                srp=False,
                drag=False,
                active_third_bodies=lambda: (),
            ),
        )
