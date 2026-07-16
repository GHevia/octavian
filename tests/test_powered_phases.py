from __future__ import annotations

import numpy as np
import pytest

from octavian import Dynamics, Mission, Phase, Spacecraft, Thruster, guesses, objectives
from octavian.dynamics import ChemicalBurnECI, FiniteThrustECI
from octavian.runner import _is_composable_mission
from octavian.solvers.compiler.phase_compiler import (
    is_powered_phase,
    mass_state_phase_indices,
    powered_phase_kind,
    prepare_phase_guess,
    validate_powered_phase_chain,
)
from octavian.solvers.compiler.powered_guessing import (
    build_low_thrust_spiral_seed,
    circular_spiral_delta_v_mps,
    constant_throttle_burn_time_s,
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
        ("low-thrust", "low_thrust"),
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


def test_low_thrust_spiral_guess_configuration_is_validated() -> None:
    config = guesses.low_thrust_spiral(
        throttle=0.75,
        direction=" RETROGRADE ",
        steps_per_orbit=80,
        time_scale=1.1,
    )

    assert config.throttle == pytest.approx(0.75)
    assert config.direction == "retrograde"
    with pytest.raises(ValueError, match="throttle"):
        guesses.low_thrust_spiral(throttle=0.0)
    with pytest.raises(ValueError, match="direction"):
        guesses.low_thrust_spiral(direction="normal")
    with pytest.raises(ValueError, match="time_scale"):
        guesses.low_thrust_spiral(time_scale=float("nan"))


def test_compiled_low_thrust_guess_rows_are_preserved() -> None:
    phase = _phase("spiral", "low_thrust", _spacecraft())
    rows = [
        np.arange(11, dtype=float),
        np.arange(11, dtype=float) + 1.0,
    ]

    prepared, layout, kind = prepare_phase_guess(phase, rows)

    assert layout.name == "cartesian_mass_thrust"
    assert kind == "low_thrust"
    np.testing.assert_allclose(prepared, rows)


def test_low_thrust_spiral_estimates_and_integrates_powered_rows() -> None:
    mu = 3.986004418e14
    radius0 = 7_000_000.0
    target_radius = 7_100_000.0
    mass0 = 550.0
    thrust = 2.0
    isp = 1_800.0
    throttle = 0.8
    delta_v = circular_spiral_delta_v_mps(radius0, target_radius, mu)
    burn_time = constant_throttle_burn_time_s(
        delta_v,
        initial_mass_kg=mass0,
        thrust_N=thrust,
        isp_s=isp,
        throttle=throttle,
    )

    rows, info = build_low_thrust_spiral_seed(
        initial_position_m=[radius0, 0.0, 0.0],
        initial_velocity_mps=[0.0, (mu / radius0) ** 0.5, 0.0],
        target_radius_m=target_radius,
        mu_m3ps2=mu,
        initial_mass_kg=mass0,
        dry_mass_kg=500.0,
        thrust_N=thrust,
        isp_s=isp,
        time_bounds_s=(0.8 * burn_time, 1.2 * burn_time),
        npts=31,
        config=guesses.low_thrust_spiral(throttle=throttle, steps_per_orbit=80),
    )

    trajectory = np.asarray(rows)
    assert trajectory.shape == (31, 11)
    assert np.all(np.diff(trajectory[:, 7]) > 0.0)
    assert np.all(np.diff(trajectory[:, 6]) < 0.0)
    assert np.max(np.linalg.norm(trajectory[:, 8:11], axis=1)) == pytest.approx(throttle)
    assert np.linalg.norm(trajectory[-1, 0:3]) > radius0
    assert info["guess_kind"] == "low_thrust_tangential_spiral"
    assert info["seed_direction"] == "prograde"
