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
