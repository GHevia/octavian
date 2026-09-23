from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("asset_asrl", exc_type=ImportError)

from octavian import state, two_burn_rendezvous
from octavian.solvers import SolverOptions, preconfigured
from octavian.specs import TwoImpulsePreCoastSpec

MU = 3.986004418e14
OPTIONS = SolverOptions(print_level=0, max_ls_iters=5, asset_threads=(1, 1))


def _boundary_states(radius_m: float):
    return (
        state([7_000e3, 0.0, 0.0], [0.0, np.sqrt(MU / 7_000e3), 0.0]),
        state([-radius_m, 0.0, 0.0], [0.0, -np.sqrt(MU / radius_m), 0.0]),
    )


@pytest.mark.parametrize("radius_m", np.linspace(8_000e3, 14_000e3, 7))
def test_quick_batch_with_zero_precoast_lower_bound(radius_m: float) -> None:
    initial, target = _boundary_states(radius_m)
    solution = two_burn_rendezvous(
        initial,
        target,
        precoast=True,
        t1_bounds_s=(0.0, 6_000.0),
        tf_bounds_s=(600.0, 20_000.0),
        nsegs=50,
        lambert_grid_size=45,
        nrevs_to_try=(0,),
        solver_options=OPTIONS,
    ).solve()

    assert solution.ok
    result = solution.result
    assert result is not None
    assert 0.0 < result.info["seed_precoast_s"] <= 6_000.0
    assert 600.0 <= result.info["seed_precoast_s"] + result.info["seed_dt_s"] <= 20_000.0
    assert 0.0 < result.info["t1_sol_s"] < result.tf_s() <= 20_000.0
    assert np.allclose(result.traj[-1, :3], target.r_m, atol=1.0)
    assert len(result.maneuvers) == 2

    # These coplanar circular transfers have a known minimum delta-v even
    # when the free departure time adds a coast around the initial orbit.
    transfer_a = 0.5 * (7_000e3 + radius_m)
    expected_dv = (
        np.sqrt(MU * (2.0 / 7_000e3 - 1.0 / transfer_a))
        - np.sqrt(MU / 7_000e3)
        + np.sqrt(MU / radius_m)
        - np.sqrt(MU * (2.0 / radius_m - 1.0 / transfer_a))
    )
    assert result.total_dv_mps() == pytest.approx(expected_dv, abs=1.0)


def test_precoast_seeds_never_extend_final_time_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    initial, target = _boundary_states(9_000e3)
    spec = TwoImpulsePreCoastSpec(
        x0=initial,
        xf=target,
        t1_bounds_s=(1.0, 20_000.0),
        tf_bounds_s=(600.0, 7_000.0),
        limit_precoast_to_one_period=False,
        min_dt_transfer_s=10.0,
        dv_front=False,
        nrevs_to_try=(0,),
    )
    propagate = preconfigured.propagate_cartesian_rv
    select_seed = preconfigured.select_best_lambert_seed
    departures = []
    windows = []

    def record_departure(rv, dt, mu):
        departures.append(dt)
        return propagate(rv, dt, mu)

    def check_window(**kwargs):
        departure = departures[-1]
        assert departure >= spec.min_dt_precoast_s
        assert kwargs["tmin_s"] >= spec.min_dt_transfer_s
        assert departure + kwargs["tmin_s"] >= spec.tf_bounds_s[0]
        assert departure + kwargs["tmax_s"] <= spec.tf_bounds_s[1]
        windows.append((kwargs["tmin_s"], kwargs["tmax_s"]))
        return select_seed(**kwargs)

    monkeypatch.setattr(preconfigured, "propagate_cartesian_rv", record_departure)
    monkeypatch.setattr(preconfigured, "select_best_lambert_seed", check_window)
    result = preconfigured.solve_two_impulse_precoast(spec, options=OPTIONS)

    assert windows
    assert result.converged
    assert result.tf_s() <= spec.tf_bounds_s[1] + 1e-3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_dt_precoast_s": 0.0}, "finite and positive"),
        ({"min_dt_transfer_s": float("nan")}, "finite and positive"),
        ({"min_dt_precoast_s": 2000.0}, "no room"),
        ({"min_dt_transfer_s": 8000.0}, "no room"),
    ],
)
def test_precoast_rejects_impossible_minimum_durations(overrides, message) -> None:
    initial, target = _boundary_states(9_000e3)
    spec = replace(TwoImpulsePreCoastSpec(x0=initial, xf=target), **overrides)
    with pytest.raises(ValueError, match=message):
        preconfigured.solve_two_impulse_precoast(spec, options=OPTIONS)
