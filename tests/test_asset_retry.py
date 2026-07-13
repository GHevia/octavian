from __future__ import annotations

import pytest

from octavian._asset import (
    AssetNonMonotonicTimeError,
    is_non_monotonic_time_error,
    solve_with_standard_sequence,
)


class _FakeTarget:
    def __init__(self) -> None:
        self.adaptive_mesh_values: list[bool] = []

    def setAdaptiveMesh(self, enabled: bool) -> None:  # noqa: N802 - mirrors ASSET API
        self.adaptive_mesh_values.append(bool(enabled))


class _FakeOCP(_FakeTarget):
    def __init__(self, failures: list[BaseException], final_result: bool = True) -> None:
        super().__init__()
        self.failures = list(failures)
        self.final_result = bool(final_result)
        self.solve_calls = 0

    def solve_optimize_solve(self) -> bool:
        self.solve_calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return self.final_result


def test_non_monotonic_time_error_detection_uses_asset_message() -> None:
    assert is_non_monotonic_time_error("Non monotonic time coordinates in LGLInterpTable.")
    assert is_non_monotonic_time_error("non-monotonic time coordinates in phase")
    assert is_non_monotonic_time_error("Non monotonic time coordinates in phase")
    assert not is_non_monotonic_time_error("line search failed")


def test_solve_retries_non_monotonic_time_with_adaptive_mesh_disabled() -> None:
    ocp = _FakeOCP([RuntimeError("non-monotonic time coordinates")])
    phase = _FakeTarget()

    converged = solve_with_standard_sequence(ocp, phases=(phase,))

    assert converged is True
    assert ocp.solve_calls == 2
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
