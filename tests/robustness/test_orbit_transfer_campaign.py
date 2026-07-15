from __future__ import annotations

import os

import pytest

pytest.importorskip("asset_asrl", exc_type=ImportError)

from .orbit_transfers import (
    build_transfer_mission,
    generate_transfer_scenarios,
    solution_checks,
)

CI_CASE_COUNT = int(os.environ.get("OCTAVIAN_ROBUSTNESS_CASES", "6"))
CAMPAIGN_SEED = int(os.environ.get("OCTAVIAN_ROBUSTNESS_SEED", "20260714"))
SCENARIOS = generate_transfer_scenarios(CI_CASE_COUNT, seed=CAMPAIGN_SEED)


@pytest.mark.robustness
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_generated_orbit_transfer_converges_and_meets_boundary(scenario) -> None:
    mission = build_transfer_mission(scenario)
    solution = mission.solve()

    metrics = solution_checks(scenario, solution)
    assert metrics["converged"] is True
