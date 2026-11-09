from __future__ import annotations

import numpy as np
import pytest

from octavian import guesses


def test_explicit_trajectory_guess_validates_dense_history() -> None:
    rows = np.zeros((3, 7), dtype=float)
    rows[:, 6] = [0.0, 1.0, 2.0]

    guess = guesses.trajectory(rows)

    np.testing.assert_array_equal(guess.rows, rows)
    rows[0, 0] = 1.0
    assert guess.rows[0, 0] == 0.0

    with pytest.raises(ValueError, match="strictly increasing"):
        guesses.trajectory(np.zeros((2, 7)))


def test_powered_trajectory_guess_uses_time_instead_of_decreasing_mass():
    from octavian import Dynamics, Phase, Spacecraft, Thruster
    from octavian.solvers.compiler.phase_compiler import prepare_phase_guess
    from octavian.solvers.composable import _explicit_trajectory_guesses

    rows = np.zeros((3, 11))
    rows[:, 0] = 7e6
    rows[:, 4] = 7500
    rows[:, 6] = [23, 22.5, 22]
    rows[:, 7] = [0, 100, 200]
    rows[:, 8:11] = [[0, 0.8, 0], [0, 0.5, 0.4], [0, 0, 0.1]]
    guess = guesses.trajectory(rows, time_column=7)
    phase = Phase(
        name="warm_start",
        mode="low_thrust",
        dynamics=Dynamics(),
        spacecraft=Spacecraft(
            dry_mass_kg=20, thrusters=[Thruster(thrust_N=0.1, isp_s=1800, propellant_mass_kg=3)]
        ),
        initial_guess=guess,
    )
    compiled, info = _explicit_trajectory_guesses([phase])
    prepared, _, _ = prepare_phase_guess(phase, compiled[0])
    np.testing.assert_array_equal(prepared, rows)
    assert info[0]["seed_duration"] == 200
    assert guess.time_column == 7
    rows[0, 8] = 1
    assert guess.rows[0, 8] == 0
    phase.initial_guess = guesses.trajectory(rows[:, :9], time_column=7)
    with pytest.raises(ValueError, match="state/time/control layout"):
        _explicit_trajectory_guesses([phase])


@pytest.mark.parametrize("time_column", [-1, 11, True, 7.0])
def test_trajectory_guess_rejects_invalid_time_column(time_column):
    with pytest.raises(ValueError, match="time_column"):
        guesses.trajectory(np.zeros((2, 11)), time_column=time_column)


def test_solution_trajectory_guess_preserves_full_native_phase_history():
    from octavian.solution import Solution
    from octavian.solvers.preconfigured import RendezvousResult

    rows = np.zeros((3, 11))
    rows[:, 6] = [23, 22.5, 22]
    rows[:, 7] = [0, 100, 200]
    rows[:, 8:11] = [[0, 0.8, 0], [0.2, 0.5, 0.4], [0, 0, 0.1]]
    result = RendezvousResult(
        converged=True,
        traj=rows[:, [0, 1, 2, 3, 4, 5, 7]],
        info={"phase_solver_trajectories": [rows.tolist()], "phase_time_columns": [7]},
    )
    solution = Solution(ok=True, result=result)
    guess = solution.trajectory_guess()
    np.testing.assert_array_equal(guess.rows, rows)
    assert guess.time_column == 7
    guess.rows[0, 6] = 10
    assert solution.trajectory_guess().rows[0, 6] == 23
    with pytest.raises(ValueError, match="phase_index"):
        solution.trajectory_guess(1)
    with pytest.raises(ValueError, match="successful"):
        Solution(ok=False, result=result).trajectory_guess()
    result.info.clear()
    with pytest.raises(ValueError, match="reusable"):
        solution.trajectory_guess()
