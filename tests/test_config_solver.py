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


def test_higher_order_collocation_applies_to_quick_transfers():
    import numpy as np

    from octavian import state, two_burn_rendezvous
    from octavian.solvers import SolverOptions

    mu = 3.986004418e14
    initial = state([7e6, 0, 0], [0, np.sqrt(mu / 7e6), 0])
    target = state([-12e6, 0, 0], [0, -np.sqrt(mu / 12e6), 0])
    mission = two_burn_rendezvous(
        initial,
        target,
        mu_m3ps2=mu,
        tf_bounds_s=(3000, 7000),
        nsegs=30,
        lambert_grid_size=20,
        nrevs_to_try=(0,),
        solver_options=SolverOptions(
            transcription_mode="LGL7", enable_adaptive_mesh=False, asset_threads=(1, 1)
        ),
    )
    solution = mission.solve()
    assert solution.ok
    assert solution.traj.shape[0] == 91
    assert solution.result.info["transcription_mode"] == "LGL7"
    assert solution.result.total_dv_mps() == pytest.approx(1751.102, rel=0.02)
