from __future__ import annotations

import numpy as np
import pytest

from octavian import EARTH, Dynamics, Perturbations, state
from octavian.astro import classical_to_cartesian
from octavian.coordinates import RELATIVE_CARTESIAN, lvlh, ric
from octavian.relative import (
    ClohessyWiltshire,
    absolute_to_relative_history,
    absolute_to_relative_orbital_elements,
    cwh_derivative,
    cwh_rendezvous_velocity,
    cwh_state_transition,
    inertial_to_relative_state,
    propagate_cwh,
    propagate_relative_numerical,
    relative_orbital_elements_to_absolute_state,
    relative_to_absolute_history,
    relative_to_inertial_state,
    ric_basis,
    select_cwh_rendezvous_seed,
)


def test_cwh_model_from_circular_orbit_provides_frame_and_scaling() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    model = ClohessyWiltshire.from_circular_orbit(
        radius_m,
        chief_name="ISS",
        reference_length_m=2_000.0,
    )

    assert model.mean_motion_radps == pytest.approx(np.sqrt(EARTH.mu_m3ps2 / radius_m**3))
    assert model.frame == lvlh("ISS")
    assert model.scaling.length_m == 2_000.0
    assert model.scaling.time_s == pytest.approx(1.0 / model.mean_motion_radps)
    assert model.scaling.velocity_mps == pytest.approx(
        model.mean_motion_radps * 2_000.0
    )

    dynamics = Dynamics.cwh(
        chief_orbit_radius_m=radius_m,
        chief_name="ISS",
        reference_length_m=2_000.0,
    )
    assert isinstance(dynamics.model, ClohessyWiltshire)
    assert dynamics.frame == lvlh("ISS")
    assert dynamics.scaling == dynamics.model.scaling
    assert dynamics.central_body is EARTH


def test_relative_layout_has_semantic_position_velocity_groups() -> None:
    assert RELATIVE_CARTESIAN.name == "relative_cartesian"
    assert RELATIVE_CARTESIAN.state_indices("position") == (0, 1, 2)
    assert RELATIVE_CARTESIAN.state_indices("velocity") == (3, 4, 5)
    assert ric("ISS").orientation == "RIC/RTN/LVLH"


def test_cwh_state_transition_matches_derivative_and_semigroup() -> None:
    n = 0.0011
    state0 = np.array([150.0, -700.0, 40.0, 0.1, -0.05, 0.02])
    dt = 1e-3
    finite_difference = (propagate_cwh(state0, dt, n) - state0) / dt
    assert finite_difference == pytest.approx(cwh_derivative(state0, n), rel=2e-5, abs=2e-8)

    phi_1 = cwh_state_transition(300.0, n)
    phi_2 = cwh_state_transition(500.0, n)
    assert cwh_state_transition(800.0, n) == pytest.approx(phi_2 @ phi_1, abs=1e-10)


def test_cwh_rendezvous_velocity_hits_requested_position() -> None:
    n = 0.0011
    r0 = np.array([0.0, -1_000.0, 100.0])
    rf = np.array([10.0, -5.0, 0.0])
    tof_s = 1_800.0
    velocity = cwh_rendezvous_velocity(r0, rf, tof_s, n)
    final_state = propagate_cwh(np.hstack([r0, velocity]), tof_s, n)
    assert final_state[0:3] == pytest.approx(rf, abs=1e-9)


def test_cwh_seed_selection_prefers_accepted_geometry_candidate() -> None:
    initial = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    final = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])
    seed = select_cwh_rendezvous_seed(
        initial,
        final,
        mean_motion_radps=0.0011,
        tof_bounds_s=(1_200.0, 2_400.0),
        samples=20,
        candidate_filter=lambda candidate: candidate.tof_s <= 1_400.0,
    )
    assert seed.tof_s <= 1_400.0


def test_inertial_relative_state_transform_round_trip() -> None:
    chief = state([7_000_000.0, 0.0, 0.0], [0.0, 7_500.0, 0.0])
    relative = state([100.0, -250.0, 40.0], [0.2, -0.1, 0.05])
    deputy = relative_to_inertial_state(chief, relative)
    recovered = inertial_to_relative_state(chief, deputy)

    assert recovered.r_m == pytest.approx(relative.r_m, abs=1e-10)
    assert recovered.v_mps == pytest.approx(relative.v_mps, abs=1e-12)


def test_ric_basis_and_history_transforms_round_trip() -> None:
    chief = np.asarray(
        [
            [7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0, 0.0],
            [6_999_000.0, 100_000.0, 0.0, -107.0, 7_499.0, 2.0, 10.0],
        ]
    )
    relative = np.asarray(
        [
            [100.0, -250.0, 40.0, 0.2, -0.1, 0.05, 0.0],
            [102.0, -251.0, 40.5, 0.21, -0.09, 0.04, 10.0],
        ]
    )
    deputy = relative_to_absolute_history(chief, relative)
    recovered = absolute_to_relative_history(chief, deputy)

    np.testing.assert_allclose(recovered, relative, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        ric_basis(chief[0, 0:3], chief[0, 3:6]),
        np.eye(3),
        atol=1e-12,
    )


def test_history_transforms_reject_mismatched_times() -> None:
    chief = np.zeros((2, 7))
    deputy = np.zeros((2, 7))
    chief[:, 0] = 7_000_000.0
    chief[:, 4] = 7_500.0
    deputy[:] = chief
    deputy[1, 6] = 2.0

    with pytest.raises(ValueError, match="time columns"):
        absolute_to_relative_history(chief, deputy)


def test_relative_orbital_elements_cartesian_round_trip() -> None:
    chief_position, chief_velocity = classical_to_cartesian(
        a_m=7_000_000.0,
        e=0.01,
        inc_deg=45.0,
        raan_deg=30.0,
        argp_deg=20.0,
        true_anomaly_deg=40.0,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    deputy_position, deputy_velocity = classical_to_cartesian(
        a_m=7_003_000.0,
        e=0.011,
        inc_deg=45.05,
        raan_deg=30.04,
        argp_deg=19.9,
        true_anomaly_deg=40.2,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    chief = state(chief_position, chief_velocity)
    deputy = state(deputy_position, deputy_velocity)

    relative_elements = absolute_to_relative_orbital_elements(
        chief,
        deputy,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    recovered = relative_orbital_elements_to_absolute_state(
        chief,
        relative_elements,
        mu_m3ps2=EARTH.mu_m3ps2,
    )

    np.testing.assert_allclose(recovered.r_m, deputy.r_m, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(recovered.v_mps, deputy.v_mps, rtol=0.0, atol=1e-9)


def test_coupled_relative_propagation_preserves_identical_states() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    speed_mps = np.sqrt(EARTH.mu_m3ps2 / radius_m)
    chief = state([radius_m, 0.0, 0.0], [0.0, speed_mps, 0.0])
    relative = state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    result = propagate_relative_numerical(
        chief,
        relative,
        np.linspace(0.0, 120.0, 5),
        perturbations=Perturbations(j2=True),
        max_step_s=2.0,
    )

    np.testing.assert_allclose(result.relative_states_ric, 0.0, atol=1e-8)
    assert result.relative_trajectory_ric.shape == (5, 7)
    assert result.chief_trajectory_eci.shape == (5, 7)


def test_j2_changes_coupled_relative_propagation() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    speed_mps = np.sqrt(EARTH.mu_m3ps2 / radius_m)
    chief = state([radius_m, 0.0, 0.0], [0.0, speed_mps, 0.0])
    relative = state([1_000.0, -500.0, 200.0], [0.0, 0.0, 0.0])
    times = np.asarray([0.0, 600.0])

    unperturbed = propagate_relative_numerical(
        chief,
        relative,
        times,
        max_step_s=2.0,
    )
    perturbed = propagate_relative_numerical(
        chief,
        relative,
        times,
        perturbations=Perturbations(j2=True),
        max_step_s=2.0,
    )

    assert np.linalg.norm(
        perturbed.relative_states_ric[-1] - unperturbed.relative_states_ric[-1]
    ) > 1e-4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mean_motion_radps": 0.0}, "mean_motion"),
        ({"mean_motion_radps": 0.001, "reference_length_m": 0.0}, "reference_length"),
    ],
)
def test_cwh_model_rejects_nonphysical_configuration(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ClohessyWiltshire(**kwargs)
