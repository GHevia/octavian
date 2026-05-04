from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from octavian import constraints
from octavian.solvers import composable

MU = 3.986004418e14


def _classical_to_cartesian(
    *,
    a_m: float,
    e: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
    mu_m3ps2: float,
) -> tuple[np.ndarray, np.ndarray]:
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    argp = np.deg2rad(argp_deg)
    nu = np.deg2rad(true_anomaly_deg)

    p = a_m * (1.0 - e**2)
    r_pf = (p / (1.0 + e * np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0.0], dtype=float)
    v_pf = np.sqrt(mu_m3ps2 / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0], dtype=float)

    cO = np.cos(raan)
    sO = np.sin(raan)
    ci = np.cos(inc)
    si = np.sin(inc)
    cw = np.cos(argp)
    sw = np.sin(argp)
    rot = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ],
        dtype=float,
    )
    return rot @ r_pf, rot @ v_pf


class _FakeVector:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=float).reshape(3)

    def cross(self, other: "_FakeVector") -> "_FakeVector":
        return _FakeVector(np.cross(self.values, other.values))

    def dot(self, other: "_FakeVector") -> float:
        return float(np.dot(self.values, other.values))

    def norm(self) -> float:
        return float(np.linalg.norm(self.values))

    def normalized(self) -> "_FakeVector":
        return _FakeVector(self.values / self.norm())

    def __getitem__(self, idx: int) -> float:
        return float(self.values[idx])


class _FakeArgs:
    def __init__(self, r_m: np.ndarray, v_mps: np.ndarray) -> None:
        self._r = _FakeVector(r_m)
        self._v = _FakeVector(v_mps)

    def tolist(self, slices):  # type: ignore[no-untyped-def]
        assert slices == [(0, 3), (3, 3)]
        return self._r, self._v


class _FakeAssetPhase:
    def __init__(self) -> None:
        self.equal_cons: list[tuple[str, float, tuple[int, ...]]] = []
        self.ineq_cons: list[tuple[str, float, tuple[int, ...]]] = []

    def addEqualCon(self, where, expr, state_indices):  # type: ignore[no-untyped-def]
        self.equal_cons.append((where, float(expr), tuple(int(i) for i in state_indices)))

    def addInequalCon(self, where, expr, state_indices):  # type: ignore[no-untyped-def]
        self.ineq_cons.append((where, float(expr), tuple(int(i) for i in state_indices)))


def test_orbital_constraint_factories_normalize_and_store_values() -> None:
    sma = constraints.semi_major_axis(8_200e3, where="back", tol_m=250.0)
    ecc = constraints.eccentricity(0.12, where="path", tol=0.01)
    inc = constraints.inclination_deg(28.5, where="front", tol_deg=0.25)

    assert sma.where == "Back"
    assert sma.value == {"a_m": 8_200e3, "tol_m": 250.0}
    assert ecc.where == "Path"
    assert ecc.value == {"e": 0.12, "tol": 0.01}
    assert inc.where == "Front"
    assert inc.value == {"inc_deg": 28.5, "tol_deg": 0.25}


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (constraints.semi_major_axis, {"a_m": 0.0}, "a_m must be non-zero"),
        (constraints.eccentricity, {"e": 0.0}, "requires e > 0"),
        (constraints.eccentricity, {"e": 0.1, "tol": 0.1}, "tol must be smaller than e"),
        (constraints.inclination_deg, {"inc_deg": 0.0}, "requires 0 < inc_deg < 180"),
        (constraints.inclination_deg, {"inc_deg": 20.0, "tol_deg": 20.0}, "tol_deg is too large"),
    ],
)
def test_orbital_constraint_factories_reject_invalid_usage(factory, kwargs, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


def test_compiler_adds_exact_orbital_element_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    r_m, v_mps = _classical_to_cartesian(
        a_m=8_400e3,
        e=0.18,
        inc_deg=28.5,
        raan_deg=40.0,
        argp_deg=15.0,
        true_anomaly_deg=55.0,
        mu_m3ps2=MU,
    )
    monkeypatch.setattr(
        composable,
        "vf",
        SimpleNamespace(Arguments=lambda _: _FakeArgs(r_m, v_mps), stack=lambda values: float(values[0])),
        raising=True,
    )

    phase = _FakeAssetPhase()
    composable._apply_orbital_element_constraint(phase, constraints.semi_major_axis(8_400e3, where="Back"), MU)
    composable._apply_orbital_element_constraint(phase, constraints.eccentricity(0.18, where="Back"), MU)
    composable._apply_orbital_element_constraint(phase, constraints.inclination_deg(28.5, where="Back"), MU)

    assert len(phase.equal_cons) == 3
    assert not phase.ineq_cons
    for where, expr, state_indices in phase.equal_cons:
        assert where == "Back"
        assert state_indices == (0, 1, 2, 3, 4, 5)
        assert abs(expr) < 1e-8


def test_compiler_adds_tolerance_bands_for_orbital_element_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r_m, v_mps = _classical_to_cartesian(
        a_m=8_900e3,
        e=0.22,
        inc_deg=32.0,
        raan_deg=10.0,
        argp_deg=70.0,
        true_anomaly_deg=25.0,
        mu_m3ps2=MU,
    )
    monkeypatch.setattr(
        composable,
        "vf",
        SimpleNamespace(Arguments=lambda _: _FakeArgs(r_m, v_mps), stack=lambda values: float(values[0])),
        raising=True,
    )

    phase = _FakeAssetPhase()
    composable._apply_orbital_element_constraint(
        phase,
        constraints.semi_major_axis(8_900e3, where="Path", tol_m=500.0),
        MU,
    )
    composable._apply_orbital_element_constraint(
        phase,
        constraints.eccentricity(0.22, where="Path", tol=0.02),
        MU,
    )
    composable._apply_orbital_element_constraint(
        phase,
        constraints.inclination_deg(32.0, where="Path", tol_deg=0.5),
        MU,
    )

    assert not phase.equal_cons
    assert len(phase.ineq_cons) == 6
    for where, expr, state_indices in phase.ineq_cons:
        assert where == "Path"
        assert state_indices == (0, 1, 2, 3, 4, 5)
        assert expr <= 1e-9
