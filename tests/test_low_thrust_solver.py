from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("asset_asrl", exc_type=ImportError)


ROOT = Path(__file__).resolve().parents[1]


def test_low_thrust_orbit_raise_converges_from_spiral_seed() -> None:
    namespace = runpy.run_path(
        str(ROOT / "examples/composable/13_low_thrust_orbit_raise.py")
    )
    mission = namespace["mission"]
    solution = mission.solve()

    assert solution.ok
    assert solution.result is not None
    result = solution.result
    assert result.info["state_layouts"] == ["cartesian_mass_thrust"]
    seed = result.info["phase_guess_info"][0]
    assert seed["guess_kind"] == "low_thrust_tangential_spiral"
    assert seed["seed_direction"] == "prograde"
    assert seed["seed_final_radius_m"] == pytest.approx(8_000_000.0, abs=25_000.0)

    powered = result.info["powered_phases"]
    assert len(powered) == 1
    assert powered[0]["kind"] == "low_thrust"
    assert 10.0 < powered[0]["propellant_used_kg"] < 25.0
    assert result.last_obj == pytest.approx(
        -powered[0]["mass_final_kg"] / mission.phases[0].spacecraft.initial_mass_kg,
        rel=1e-8,
    )
    assert all(row["satisfied"] for row in result.info["constraint_report"])
