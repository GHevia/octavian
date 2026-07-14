from __future__ import annotations

import pytest

from octavian import Dynamics, Mission, Phase, Spacecraft, Thruster, objectives
from octavian.dynamics import ChemicalBurnECI, FiniteThrustECI
from octavian.runner import _is_composable_mission
from octavian.solvers.compiler.phase_compiler import (
    is_powered_phase,
    mass_state_phase_indices,
    powered_phase_kind,
    validate_powered_phase_chain,
)


def _spacecraft(name: str = "vehicle") -> Spacecraft:
    return Spacecraft(
        name=name,
        dry_mass_kg=500.0,
        thrusters=[
            Thruster(
                name="main",
                thrust_N=2_000.0,
                isp_s=320.0,
                propellant_mass_kg=50.0,
            )
        ],
    )


def _phase(name: str, mode: str, spacecraft: Spacecraft) -> Phase:
    return Phase(
        name=name,
        mode=mode,
        spacecraft=spacecraft,
        dynamics=Dynamics(),
    )


def test_finite_thrust_is_the_neutral_dynamics_name() -> None:
    assert ChemicalBurnECI is FiniteThrustECI


@pytest.mark.parametrize(
    ("mode", "kind"),
    [
        ("chemical_burn", "chemical_burn"),
        ("finite-burn", "chemical_burn"),
        ("finite_thrust", "finite_thrust"),
        ("powered", "finite_thrust"),
    ],
)
def test_powered_phase_modes_are_normalized(mode: str, kind: str) -> None:
    phase = _phase("burn", mode, _spacecraft())

    assert powered_phase_kind(phase) == kind
    assert is_powered_phase(phase)
    assert _is_composable_mission(Mission(phases=[phase]))


def test_mass_is_carried_only_between_first_and_last_powered_phase() -> None:
    spacecraft = _spacecraft()
    phases = [
        _phase("precoast", "coast", spacecraft),
        _phase("departure", "finite_thrust", spacecraft),
        _phase("transfer", "coast", spacecraft),
        _phase("trim", "powered", spacecraft),
        _phase("postcoast", "coast", spacecraft),
    ]

    validate_powered_phase_chain(phases)
    assert mass_state_phase_indices(phases) == {1, 2, 3}


def test_powered_chain_rejects_ambiguous_spacecraft_changes() -> None:
    first = _spacecraft("first")
    second = _spacecraft("second")
    phases = [
        _phase("departure", "finite_thrust", first),
        _phase("transfer", "coast", first),
        _phase("arrival", "finite_thrust", second),
    ]

    with pytest.raises(ValueError, match="one Spacecraft configuration"):
        validate_powered_phase_chain(phases)


def test_propellant_objective_is_explicit() -> None:
    objective = objectives.minimize_propellant(weight=2.0)

    assert objective.kind == "propellant"
    assert objective.weight == pytest.approx(2.0)
