from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from octavian import Mission, Phase, Spacecraft, Thruster, constraints, objectives, state
from octavian.models import Dynamics, Perturbations
from octavian.runner import _is_composable_mission
from octavian.solvers import composable

MU = 3.986004418e14
ROOT = Path(__file__).resolve().parents[1]


def _burn_phase() -> Phase:
    spacecraft = Spacecraft(
        name="burner",
        dry_mass_kg=500.0,
        thrusters=[Thruster(name="main", thrust_N=2_000.0, isp_s=320.0, propellant_mass_kg=50.0)],
    )
    x0 = state([7000e3, 0.0, 0.0], [0.0, np.sqrt(MU / 7000e3), 0.0])
    xf = state([7100e3, 20e3, 0.0], [10.0, np.sqrt(MU / 7100e3), 0.0])
    return Phase(
        name="injection",
        mode="chemical_burn",
        spacecraft=spacecraft,
        dynamics=Dynamics(mu_m3ps2=MU, j2=True),
        initial_state=x0,
        final_state=xf,
        tof_bounds_s=(10.0, 500.0),
        constraints=[constraints.state(x0, where="Front"), constraints.state(xf, where="Back")],
    )


def _burn_coast_burn_phases() -> list[Phase]:
    burn0 = _burn_phase()
    burn0.name = "departure_burn"
    burn0.constraints = [constraints.state(burn0.initial_state, where="Front")]
    burn0.final_state = None

    coast = Phase(
        name="coast",
        mode="coast",
        spacecraft=burn0.spacecraft,
        dynamics=burn0.dynamics,
        previous=burn0,
        tof_bounds_s=(1_000.0, 4_000.0),
        tof_is_relative=True,
    )

    burn1 = _burn_phase()
    burn1.name = "arrival_burn"
    burn1.previous = coast
    burn1.initial_state = None
    burn1.tof_bounds_s = (10.0, 500.0)
    burn1.tof_is_relative = True
    return [burn0, coast, burn1]


def test_chemical_burn_mode_uses_composable_backend() -> None:
    mission = Mission(phases=[_burn_phase()])

    assert _is_composable_mission(mission) is True
    assert composable._phase_is_chemical_burn(mission.phases[0]) is True
    assert composable._phase_dimensions(mission.phases[0]) == (7, 3, True)


def test_chemical_burn_transfer_requires_burn_coast_burn_shape() -> None:
    with pytest.raises(ValueError, match="departure burn, a coast, and an arrival burn"):
        composable._validate_chemical_burn_transfer([_burn_phase()])

    phases = _burn_coast_burn_phases()
    composable._validate_chemical_burn_transfer(phases)
    assert composable._mass_state_phase_indices(phases) == {0, 1, 2}


def test_j2_perturbation_uses_composable_backend() -> None:
    phase = _burn_phase()
    phase.mode = "coast"
    phase.dynamics = Dynamics(mu_m3ps2=MU, perturbations=Perturbations(j2=True))

    assert phase.dynamics.active_perturbations().j2 is True
    assert _is_composable_mission(Mission(phases=[phase])) is True


def test_chemical_burn_guess_adds_mass_time_and_direction_controls() -> None:
    phase = _burn_phase()
    base_guess = [
        np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0, 0.0]),
        np.array([7001e3, 10e3, 0.0, 20.0, 7520.0, 0.0, 100.0]),
    ]

    rows = composable._augment_guess_for_chemical_burn(
        base_guess,
        phase=phase,
        mass0_kg=550.0,
        thrust_N=2_000.0,
        isp_s=320.0,
    )

    assert len(rows) == 2
    assert rows[0].shape == (11,)
    assert rows[0][6] == pytest.approx(550.0)
    assert rows[0][7] == pytest.approx(0.0)
    assert rows[-1][7] == pytest.approx(100.0)
    assert rows[-1][6] < rows[0][6]
    assert np.linalg.norm(rows[0][8:11]) <= 1.0
    np.testing.assert_allclose(rows[0][8:11], rows[-1][8:11])


def test_mass_coast_guess_carries_constant_mass_state() -> None:
    phase = _burn_coast_burn_phases()[1]
    phase.info["_mass_guess_start_kg"] = 540.0
    base_guess = [
        np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0, 20.0]),
        np.array([7100e3, 20e3, 0.0, 10.0, 7520.0, 0.0, 120.0]),
    ]

    rows = composable._augment_guess_for_mass_coast(
        base_guess,
        phase=phase,
        mass0_kg=550.0,
    )

    assert rows[0].shape == (8,)
    assert rows[0][6] == pytest.approx(540.0)
    assert rows[-1][6] == pytest.approx(540.0)
    assert rows[0][7] == pytest.approx(20.0)
    assert rows[-1][7] == pytest.approx(120.0)


def test_unsupported_perturbation_flags_fail_before_asset_build() -> None:
    phase = _burn_phase()
    phase.dynamics = Dynamics(mu_m3ps2=MU, perturbations=Perturbations(drag=True))

    with pytest.raises(NotImplementedError, match="J2 perturbations only"):
        composable._phase_perturbations(phase)


def test_zero_weight_objective_remains_zero() -> None:
    mission = Mission(objectives=[objectives.minimize_total_delta_v(weight=0.0)])

    minimize_dv, weight, minimize_time, time_weight = composable._objective_weights(mission)

    assert minimize_dv is True
    assert weight == pytest.approx(0.0)
    assert minimize_time is False
    assert time_weight == pytest.approx(0.0)


def test_example_08_configures_burn_coast_burn_with_j2() -> None:
    namespace = runpy.run_path(str(ROOT / "examples/composable/08_chemical_burn_j2.py"))
    mission = namespace["mission"]
    phases = mission.phases

    assert [phase.mode for phase in phases] == ["chemical_burn", "coast", "chemical_burn"]
    assert all(phase.dynamics.active_perturbations().j2 for phase in phases)
    assert mission.objectives[0].weight == pytest.approx(0.0)
    composable._validate_chemical_burn_transfer(phases)


def test_example_09_compares_impulse_and_chemical_transfers() -> None:
    namespace = runpy.run_path(str(ROOT / "examples/composable/09_impulse_vs_chemical_burn.py"))

    chemical_mission = namespace["chemical_mission"]

    assert namespace["COAST_BOUNDS_S"] == (1_800.0, 3_000.0)
    assert [phase.mode for phase in chemical_mission.phases] == [
        "chemical_burn",
        "coast",
        "chemical_burn",
    ]
    composable._validate_chemical_burn_transfer(chemical_mission.phases)
