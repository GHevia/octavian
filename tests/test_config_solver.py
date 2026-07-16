from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("asset_asrl", exc_type=ImportError)

from octavian import load_mission

ROOT = Path(__file__).resolve().parents[1]


def test_json_hohmann_mission_converges() -> None:
    mission = load_mission(ROOT / "examples/config/01_two_impulse_transfer.json")
    solution = mission.solve()

    assert solution.ok
    assert solution.result is not None
    assert solution.result.tf_s() == pytest.approx(4_607.511, rel=2.0e-2, abs=30.0)
    assert solution.result.total_dv_mps() == pytest.approx(1_751.102, rel=2.0e-2, abs=10.0)
