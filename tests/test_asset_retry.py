from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from octavian._asset import (
    AssetNonMonotonicTimeError,
    is_non_monotonic_time_error,
    set_ocp_threads,
    solve_with_standard_sequence,
)


class Flag(Enum):
    CONVERGED = 0
    ACCEPTABLE = 1
    NOTCONVERGED = 2
    DIVERGING = 3


class _FakeTarget:
    def __init__(self) -> None:
        self.adaptive_mesh_values: list[bool] = []

    def returnTraj(self):
        return [[0, 0], [1, 1]]

    def returnCostateTraj(self):
        return [[-1, 0], [-1, 1]]

    def returnTrajError(self):
        return [[0, 0], [0, 1]]

    def setAdaptiveMesh(self, enabled: bool) -> None:  # noqa: N802 - mirrors ASSET API
        self.adaptive_mesh_values.append(bool(enabled))


class _FakeOCP(_FakeTarget):
    def __init__(self, failures: list[BaseException], final_result: bool = True) -> None:
        super().__init__()
        self.failures = list(failures)
        self.final_result = Flag.CONVERGED if final_result else Flag.NOTCONVERGED
        self.optimizer = SimpleNamespace(LastIterNum=1, LastObjVal=1.0)
        self.solve_calls = 0
        self.thread_values: list[tuple[int, int]] = []

    def solve(self):
        self.solve_calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.final_result

    def optimize(self):
        return self.solve()

    def setThreads(self, optimizer_threads: int, mesh_threads: int) -> None:  # noqa: N802 - mirrors ASSET API
        self.thread_values.append((int(optimizer_threads), int(mesh_threads)))


def test_non_monotonic_time_error_detection_uses_asset_message() -> None:
    assert is_non_monotonic_time_error("Non monotonic time coordinates in LGLInterpTable.")
    assert is_non_monotonic_time_error("non-monotonic time coordinates in phase")
    assert is_non_monotonic_time_error("Non monotonic time coordinates in phase")
    assert is_non_monotonic_time_error("Duplicate time coordinates in LGLInterpTable.")
    assert not is_non_monotonic_time_error("line search failed")


def test_solve_retries_non_monotonic_time_with_adaptive_mesh_disabled() -> None:
    ocp = _FakeOCP([RuntimeError("non-monotonic time coordinates")])
    phase = _FakeTarget()

    converged = solve_with_standard_sequence(ocp, phases=(phase,))

    assert converged.feasible
    assert converged.optimality_converged
    assert ocp.solve_calls == 3
    assert ocp.adaptive_mesh_values == [False]
    assert phase.adaptive_mesh_values == [False]


def test_solve_wraps_repeated_non_monotonic_time_failures() -> None:
    ocp = _FakeOCP(
        [
            RuntimeError("non-monotonic time coordinates"),
            RuntimeError("non-monotonic time coordinates"),
        ]
    )

    with pytest.raises(AssetNonMonotonicTimeError) as excinfo:
        solve_with_standard_sequence(ocp)

    message = str(excinfo.value)
    assert "adaptive mesh disabled" in message
    assert "coarser mesh" in message
    assert ocp.solve_calls == 2


def test_solve_does_not_retry_unrelated_asset_errors() -> None:
    ocp = _FakeOCP([RuntimeError("line search failed")])

    with pytest.raises(RuntimeError, match="line search failed"):
        solve_with_standard_sequence(ocp)

    assert ocp.solve_calls == 1


def test_set_ocp_threads_configures_asset_thread_pair() -> None:
    ocp = _FakeOCP([])

    set_ocp_threads(ocp, (1, 1))

    assert ocp.thread_values == [(1, 1)]


def test_set_ocp_threads_rejects_invalid_thread_counts() -> None:
    ocp = _FakeOCP([])

    with pytest.raises(ValueError, match="positive integers"):
        set_ocp_threads(ocp, (1, 0))


def test_failed_optimization_and_feasibility_recovery_have_separate_statuses():
    ocp = _FakeOCP([])
    flags = iter([Flag.CONVERGED, Flag.NOTCONVERGED, Flag.CONVERGED])
    ocp.solve = lambda: next(flags)
    report = solve_with_standard_sequence(ocp, phases=(_FakeTarget(),))
    assert report.feasible
    assert not report.optimality_converged
    assert [row["flag"] for row in report.stages] == ["CONVERGED", "NOTCONVERGED", "CONVERGED"]
    np.testing.assert_array_equal(report.optimization_phases[0]["costates"], [[-1, 0], [-1, 1]])


def test_truthy_diverging_flag_is_not_success():
    ocp = _FakeOCP([])
    ocp.final_result = Flag.DIVERGING
    assert bool(Flag.DIVERGING)
    report = solve_with_standard_sequence(ocp)
    assert not report.feasible
    assert not report.optimality_converged


def test_native_costates_match_minimum_energy_integrator():
    from octavian._asset import oc, vf

    if oc is None:
        pytest.skip("ASSET is unavailable")
    args = oc.ODEArguments(1, 1)
    ode = oc.ODEBase(args.UVec(), 1, 1)
    phase = ode.phase("LGL5", [np.array([t, t, 1.0]) for t in (0, 0.5, 1)], 8)
    phase.addBoundaryValue("Front", [0, 1], np.array([0.0, 0.0]))
    phase.addBoundaryValue("Back", [0, 1], np.array([1.0, 1.0]))
    phase.addIntegralObjective(vf.Arguments(1)[0] ** 2 / 2, [2])
    problem = oc.OptimalControlProblem()
    problem.addPhase(phase)
    problem.optimizer.PrintLevel = 3
    problem.setThreads(1, 1)
    report = solve_with_standard_sequence(problem, phases=(phase,))
    assert report.optimality_converged and report.feasible
    np.testing.assert_allclose(
        np.asarray(report.optimization_phases[0]["costates"])[:, 0], -1, atol=1e-6
    )
    from octavian.solution import Solution
    from octavian.solvers.preconfigured import RendezvousResult

    solution = Solution(
        ok=True,
        result=RendezvousResult(
            converged=True,
            traj=np.asarray(phase.returnTraj()),
            asset_phases=(phase,),
            asset_odes=(ode,),
            info={"phase_time_columns": [1], "solve_report": report.to_dict()},
        ),
    )
    validation = solution.validate_dynamics(max_step_s=0.1, state_tolerances=[1e-6])
    assert validation[0]["within_tolerance"]
    assert solution.optimality_converged
