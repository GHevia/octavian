from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from octavian.solution import Solution
from octavian.solvers.preconfigured import RendezvousResult
from octavian.types import Maneuver

ROOT = Path(__file__).resolve().parents[1]


def _fake_solution() -> Solution:
    traj = np.zeros((3, 7), dtype=float)
    traj[:, 6] = [0.0, 50.0, 100.0]
    maneuvers = [Maneuver(r_m=[0.0, 0.0, 0.0], t_s=50.0, dv_mps=[1.0, 0.0, 0.0], name="dv")]
    res = RendezvousResult(converged=True, traj=traj, maneuvers=maneuvers, last_obj=1.0, info={})
    return Solution(ok=True, result=res)


@pytest.mark.parametrize(
    ("script_rel", "expected_solve_calls"),
    [
        ("examples/quick/01_two_impulse_free_time.py", 1),
        ("examples/quick/02_two_impulse_precoast_impulsive_link.py", 1),
        ("examples/quick/03_time_tradeoff.py", 2),
        ("examples/quick/04_batch_targets.py", 7),
        ("examples/quick/05_sun_centered_transfer.py", 1),
    ],
)
def test_quick_examples_run_as_scenarios(monkeypatch: pytest.MonkeyPatch, script_rel: str, expected_solve_calls: int) -> None:
    missions = []
    plotted = []

    def fake_solve(self):  # type: ignore[no-untyped-def]
        missions.append(self)
        return _fake_solution()

    def fake_plot(*args, **kwargs):  # type: ignore[no-untyped-def]
        if "out_html" in kwargs:
            plotted.append(kwargs["out_html"])
        elif len(args) >= 2:
            plotted.append(args[1])

    monkeypatch.setattr("octavian.mission.Mission.solve", fake_solve, raising=True)
    monkeypatch.setattr("octavian.viz.plotly.save_trajectory_html", fake_plot, raising=True)
    monkeypatch.setattr("octavian.viz.save_trajectory_html", fake_plot, raising=True)

    runpy.run_path(str(ROOT / script_rel), run_name="__main__")

    assert len(missions) == expected_solve_calls
    if script_rel.endswith("05_sun_centered_transfer.py"):
        assert plotted == []
    else:
        assert len(plotted) >= 1

    if script_rel.endswith("01_two_impulse_free_time.py"):
        assert len(missions[0].phases) == 1
        assert missions[0].phases[0].mode.lower() == "rendezvous"
    elif script_rel.endswith("02_two_impulse_precoast_impulsive_link.py"):
        assert len(missions[0].phases) == 2
        assert missions[0].phases[0].mode.lower() == "coast"
        assert missions[0].phases[1].mode.lower() == "rendezvous"
    elif script_rel.endswith("03_time_tradeoff.py"):
        assert len(missions[0].phases) == 1
        assert len(missions[1].phases) == 1
        assert float(missions[0].w_time) == 0.0
        assert float(missions[1].w_time) > 0.0
    elif script_rel.endswith("04_batch_targets.py"):
        assert all(len(m.phases) == 2 for m in missions)
    elif script_rel.endswith("05_sun_centered_transfer.py"):
        assert missions[0].phases[0].dynamics.central_body.name == "sun"
        assert missions[0].phases[0].dynamics.frame.origin == "sun"
