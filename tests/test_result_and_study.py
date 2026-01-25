from __future__ import annotations

import numpy as np

from octavian.types import Maneuver
from octavian.solvers.rendezvous import RendezvousResult
from octavian.study import best_by


def test_result_npz_roundtrip(tmp_path):
    traj = np.zeros((3, 7), dtype=float)
    traj[:, -1] = [0.0, 5.0, 10.0]
    mans = [
        Maneuver(r_m=[1, 2, 3], t_s=0.0, dv_mps=[0.1, 0.0, 0.0], name="dv1"),
        Maneuver(r_m=[4, 5, 6], t_s=10.0, dv_mps=[0.0, 0.2, 0.0], name="dv2"),
    ]
    r0 = RendezvousResult(converged=True, traj=traj, maneuvers=mans, last_obj=123.0, info={"seed": "x"})
    path = tmp_path / "case.npz"
    r0.to_npz(path)
    r1 = RendezvousResult.from_npz(path)

    assert r1.converged is True
    assert np.allclose(r1.traj, r0.traj)
    assert len(r1.maneuvers) == 2
    assert r1.maneuvers[0].name == "dv1"
    assert np.allclose(r1.maneuvers[1].dv_mps, mans[1].dv_mps)
    assert r1.info["seed"] == "x"


def test_best_by_total_dv():
    traj = np.zeros((2, 7), dtype=float)
    traj[:, -1] = [0.0, 1.0]

    r_small = RendezvousResult(
        converged=True,
        traj=traj,
        maneuvers=[Maneuver(r_m=[0, 0, 0], t_s=0.0, dv_mps=[1, 0, 0])],
    )
    r_big = RendezvousResult(
        converged=True,
        traj=traj,
        maneuvers=[Maneuver(r_m=[0, 0, 0], t_s=0.0, dv_mps=[3, 0, 0])],
    )
    best = best_by([r_big, r_small], key="total_dv_mps")
    assert best is r_small
