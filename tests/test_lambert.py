from __future__ import annotations

import numpy as np
import pytest

from octavian.astro import lambert


def _select_exact_antipodal_seed():
    return lambert.select_best_lambert_seed(
        r0_m=np.array([7_000_000.0, 0.0, 0.0]),
        rf_m=np.array([-12_000_000.0, 0.0, 0.0]),
        v0_mps=np.array([0.0, 7_500.0, 0.0]),
        vf_mps=np.array([0.0, -5_800.0, 0.0]),
        mu_m3ps2=3.986004418e14,
        tmin_s=3_000.0,
        tmax_s=7_000.0,
        n_tofs=2,
        nrevs=(0,),
    )


def test_exact_antipodal_seed_uses_a_finite_regularized_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_positions: list[np.ndarray] = []

    def fake_lambert(r0, rf, tof_s, mu, longway, nrev, rightbranch):
        target_positions.append(np.asarray(rf, dtype=float))
        return np.array([0.0, 8_000.0, 0.0]), np.array([0.0, -5_000.0, 0.0])

    monkeypatch.setattr(lambert, "_call_lambert_izzo", fake_lambert)

    seed = _select_exact_antipodal_seed()

    assert np.isfinite(seed.total_dv_mps)
    assert target_positions
    assert all(np.all(np.isfinite(target)) for target in target_positions)
    assert all(target[1] != 0.0 for target in target_positions)
    assert all(np.linalg.norm(target) == pytest.approx(12_000_000.0) for target in target_positions)


def test_non_finite_lambert_candidates_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nan_lambert(r0, rf, tof_s, mu, longway, nrev, rightbranch):
        return np.full(3, np.nan), np.full(3, np.nan)

    monkeypatch.setattr(lambert, "_call_lambert_izzo", nan_lambert)

    with pytest.raises(RuntimeError, match="No finite Lambert solution"):
        _select_exact_antipodal_seed()
