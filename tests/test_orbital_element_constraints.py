from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from octavian import constraints
from octavian.astro import classic_to_cartesian, classical_to_cartesian
from octavian.quick import state
from octavian.solvers import composable
from octavian.variables import ImpulsiveDeltaV

MU = 3.986004418e14


class _FakeVector:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=float).reshape(3)

    def cross(self, other: _FakeVector) -> _FakeVector:
        return _FakeVector(np.cross(self.values, other.values))

    def dot(self, other: _FakeVector) -> float:
        return float(np.dot(self.values, other.values))

    def norm(self) -> float:
        return float(np.linalg.norm(self.values))

    def normalized(self) -> _FakeVector:
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
    r_m, v_mps = classic_to_cartesian(
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
    r_m, v_mps = classic_to_cartesian(
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


def test_classical_to_cartesian_alias_matches_classic_to_cartesian() -> None:
    r0, v0 = classic_to_cartesian(
        a_m=8_400e3,
        e=0.18,
        inc_deg=28.5,
        raan_deg=40.0,
        argp_deg=15.0,
        true_anomaly_deg=55.0,
        mu_m3ps2=MU,
    )
    r1, v1 = classical_to_cartesian(
        a_m=8_400e3,
        e=0.18,
        inc_deg=28.5,
        raan_deg=40.0,
        argp_deg=15.0,
        true_anomaly_deg=55.0,
        mu_m3ps2=MU,
    )
    assert np.allclose(r0, r1)
    assert np.allclose(v0, v1)


def test_back_impulse_target_requires_explicit_terminal_velocity_constraint() -> None:
    x = state(r_m=[1.0, 2.0, 3.0], v_mps=[4.0, 5.0, 6.0])
    phase = SimpleNamespace(
        constraints=[],
        initial_state=None,
        final_state=x,
        variables=[ImpulsiveDeltaV(where="Back")],
    )
    assert composable._explicit_boundary_velocity_target(phase, "Back") is None


def test_back_impulse_target_uses_state_constraint_velocity_group() -> None:
    x = state(r_m=[1.0, 2.0, 3.0], v_mps=[4.0, 5.0, 6.0])
    phase = SimpleNamespace(
        constraints=[constraints.state(x, where="Back", groups=("R", "V"))],
        initial_state=None,
        final_state=None,
        variables=[ImpulsiveDeltaV(where="Back")],
    )
    assert np.allclose(composable._explicit_boundary_velocity_target(phase, "Back"), x.v_mps)
