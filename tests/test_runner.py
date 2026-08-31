from __future__ import annotations

import numpy as np
import pytest

from octavian.mission import Mission
from octavian.models import Dynamics, RetryPolicy, RunPlan, SolveConfig, Stage
from octavian.phase import Phase, state
from octavian.runner import MissionRunner
from octavian.solution import Solution
from octavian.solvers import SolverOptions, composable
from octavian.solvers.preconfigured import RendezvousResult
from octavian.solvers.rendezvous import RendezvousResult as LegacyRendezvousResult
from octavian.spacecraft import Spacecraft
from octavian.variables import ImpulsiveDeltaV


def _boundary_states():
    x0 = state([7_000_000.0, 0.0, 0.0], [0.0, 7_500.0, 0.0])
    xf = state([0.0, 7_000_000.0, 0.0], [-7_500.0, 0.0, 0.0])
    return x0, xf


def _single_phase_mission() -> Mission:
    x0, xf = _boundary_states()
    return Mission(
        phases=[
            Phase(
                name="transfer",
                mode="rendezvous",
                spacecraft=Spacecraft(name="SC"),
                dynamics=Dynamics(),
                initial_state=x0,
                final_state=xf,
                tof_bounds_s=(600.0, 1200.0),
            )
        ],
        mesh_nsegs_transfer=40,
    )


def _result(converged: bool = True) -> RendezvousResult:
    traj = np.zeros((2, 7), dtype=float)
    traj[:, 6] = [0.0, 100.0]
    return RendezvousResult(converged=converged, traj=traj)


def test_runner_scales_rendezvous_stage_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_nsegs: list[int] = []

    def fake_solve(spec, *, options=None):  # type: ignore[no-untyped-def]
        seen_nsegs.append(spec.nsegs)
        return _result()

    monkeypatch.setattr("octavian.runner.solve_preconfigured", fake_solve)
    runner = MissionRunner(
        solve_options=SolverOptions(),
        solve_config=SolveConfig(max_attempts=1),
        plan=RunPlan(stages=[Stage(name="coarse", nsegs_scale=0.5)]),
    )

    solution = runner.solve(_single_phase_mission())

    assert solution.ok is True
    assert seen_nsegs == [20]
    assert solution.attempts[0].stage == "coarse"
    assert solution.info["stage_index"] == 0


def test_legacy_rendezvous_import_path_still_exports_result_type() -> None:
    assert LegacyRendezvousResult is RendezvousResult


def test_reported_constraint_failure_overrides_optimizer_convergence() -> None:
    report = [
        {
            "constraint": "jacobi_constant",
            "target": 3.16,
            "actual": 3.15,
            "satisfied": False,
        }
    ]

    assert composable._validated_convergence(True, report) is False
    result = RendezvousResult(
        converged=False,
        traj=np.zeros((2, 7)),
        info={
            "optimizer_converged": True,
            "constraint_report": report,
        },
    )
    summary = result.summary()
    assert "Octavian result: NOT CONVERGED" in summary
    assert "reported constraint validation: FAILED" in summary
    assert "ok=False" in summary


def test_runner_retries_failed_rendezvous_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_grid_sizes: list[int] = []

    def fake_solve(spec, *, options=None):  # type: ignore[no-untyped-def]
        seen_grid_sizes.append(spec.lambert_grid_size)
        if len(seen_grid_sizes) == 1:
            raise RuntimeError("seed failed")
        return _result()

    monkeypatch.setattr("octavian.runner.solve_preconfigured", fake_solve)
    runner = MissionRunner(
        solve_options=SolverOptions(),
        solve_config=SolveConfig(max_attempts=2),
    )

    solution = runner.solve(_single_phase_mission())

    assert solution.ok is True
    assert seen_grid_sizes == [60, 80]
    assert [attempt.status for attempt in solution.attempts] == ["fail", "ok"]


def test_runner_dispatches_composable_missions(monkeypatch: pytest.MonkeyPatch) -> None:
    mission = _single_phase_mission()
    mission.phases[0].variables.append(ImpulsiveDeltaV(where="Front"))
    called = []

    def fake_composable(mission_arg, *, options=None):  # type: ignore[no-untyped-def]
        called.append(mission_arg)
        return _result()

    monkeypatch.setattr("octavian.runner.solve_composable_mission", fake_composable)

    solution = mission.solve(solve_config=SolveConfig(max_attempts=1))

    assert isinstance(solution, Solution)
    assert solution.ok is True
    assert called == [mission]


def test_runner_marks_returned_nonconverged_attempt_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "octavian.runner.solve_preconfigured",
        lambda spec, *, options=None: _result(converged=False),
    )

    solution = MissionRunner(
        solve_options=SolverOptions(),
        solve_config=SolveConfig(max_attempts=1, raise_on_fail=False),
    ).solve(_single_phase_mission())

    assert solution.ok is False
    assert solution.result is not None
    assert [attempt.status for attempt in solution.attempts] == ["fail"]
    assert "Octavian result: NOT CONVERGED" in solution.summary()


def test_retry_policy_limits_attempt_count(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_solve(spec, *, options=None):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        raise RuntimeError("still failing")

    monkeypatch.setattr("octavian.runner.solve_preconfigured", fake_solve)
    runner = MissionRunner(
        solve_options=SolverOptions(),
        solve_config=SolveConfig(max_attempts=5, raise_on_fail=False),
        retry=RetryPolicy(enabled=True, max_retries=1),
    )

    solution = runner.solve(_single_phase_mission())

    assert solution.ok is False
    assert calls == 2
    assert len(solution.attempts) == 2
