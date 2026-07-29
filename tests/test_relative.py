from __future__ import annotations

import numpy as np
import pytest

from octavian import EARTH, Dynamics, Perturbations, state
from octavian.astro import classical_to_cartesian
from octavian.coordinates import (
    CLASSICAL_RELATIVE_ELEMENTS,
    COUPLED_RELATIVE_ECI,
    COUPLED_RELATIVE_RIC,
    DAMICO_RELATIVE_ELEMENTS,
    RELATIVE_CARTESIAN,
    lvlh,
    ric,
)
from octavian.relative import (
    ClassicalRelativeOrbitalElements,
    ClohessyWiltshire,
    NonlinearRelative,
    RelativeOrbitalElements,
    RelativePropagationMode,
    absolute_to_classical_relative_orbital_elements,
    absolute_to_relative_history,
    absolute_to_relative_orbital_elements,
    classical_relative_orbital_elements_to_absolute_state,
    classical_to_damico_relative_orbital_elements,
    coupled_relative_ric_derivative,
    cwh_derivative,
    cwh_rendezvous_velocity,
    cwh_state_transition,
    damico_to_classical_relative_orbital_elements,
    inertial_to_relative_state,
    nonlinear_relative_ric_derivative,
    propagate_cwh,
    propagate_nonlinear_relative_ric,
    propagate_relative_elements_to_ric,
    propagate_relative_numerical,
    propagate_relative_orbital_elements,
    relative_orbital_elements_to_absolute_state,
    relative_orbital_elements_to_relative_state,
    relative_state_to_relative_orbital_elements,
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
    assert model.scaling.velocity_mps == pytest.approx(model.mean_motion_radps * 2_000.0)

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
    assert COUPLED_RELATIVE_ECI.state_indices("chief_position") == (0, 1, 2)
    assert COUPLED_RELATIVE_ECI.state_indices("deputy_velocity") == (9, 10, 11)
    assert COUPLED_RELATIVE_RIC.state_indices("position") == (6, 7, 8)
    assert DAMICO_RELATIVE_ELEMENTS.state_indices("delta_lambda") == (1,)
    assert CLASSICAL_RELATIVE_ELEMENTS.state_indices("delta_mean_anomaly") == (5,)


def test_nonlinear_relative_model_uses_chief_and_relative_scaling() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    chief = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )

    dynamics = Dynamics.relative(
        chief_initial_state_eci=chief,
        chief_name="ISS",
        reference_length_m=2_000.0,
    )

    assert isinstance(dynamics.model, NonlinearRelative)
    assert dynamics.frame == ric("ISS")
    assert dynamics.scaling.length_m == 2_000.0
    assert dynamics.model.propagation_mode is RelativePropagationMode.COUPLED_ECI


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("coupled_eci", RelativePropagationMode.COUPLED_ECI),
        ("stacked_ric", RelativePropagationMode.COUPLED_RIC),
        ("exact_ric", RelativePropagationMode.NONLINEAR_RIC),
        ("roe", RelativePropagationMode.DAMICO),
        ("classical", RelativePropagationMode.CLASSICAL_ELEMENTS),
    ],
)
def test_relative_model_normalizes_propagation_modes(
    requested: str,
    expected: RelativePropagationMode,
) -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    chief = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )
    dynamics = Dynamics.relative(
        chief_initial_state_eci=chief,
        propagation_mode=requested,
    )
    assert dynamics.model is not None
    assert dynamics.model.propagation_mode is expected


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


def test_accelerating_ric_state_transform_round_trip() -> None:
    chief = state([7_000_000.0, 0.0, 0.0], [0.0, 7_500.0, 200.0])
    relative = state([100.0, -250.0, 40.0], [0.2, -0.1, 0.05])
    acceleration = np.asarray([-8.0, 0.0, 0.02])

    deputy = relative_to_inertial_state(
        chief,
        relative,
        chief_acceleration_mps2=acceleration,
    )
    recovered = inertial_to_relative_state(
        chief,
        deputy,
        chief_acceleration_mps2=acceleration,
    )

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


def test_relative_element_representations_and_ric_round_trip() -> None:
    chief_position, chief_velocity = classical_to_cartesian(
        a_m=7_200_000.0,
        e=0.02,
        inc_deg=48.0,
        raan_deg=25.0,
        argp_deg=35.0,
        true_anomaly_deg=15.0,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    chief = state(chief_position, chief_velocity)
    relative = state([150.0, -400.0, 75.0], [0.04, -0.02, 0.01])

    damico = relative_state_to_relative_orbital_elements(
        chief,
        relative,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    recovered_relative = relative_orbital_elements_to_relative_state(
        chief,
        damico,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    classical = damico_to_classical_relative_orbital_elements(
        chief,
        damico,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    recovered_damico = classical_to_damico_relative_orbital_elements(
        chief,
        classical,
        mu_m3ps2=EARTH.mu_m3ps2,
    )

    np.testing.assert_allclose(recovered_relative.r_m, relative.r_m, atol=1e-6)
    np.testing.assert_allclose(
        recovered_relative.v_mps,
        relative.v_mps,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        recovered_damico.as_vector(),
        damico.as_vector(),
        atol=1e-12,
    )


def test_classical_relative_elements_absolute_round_trip() -> None:
    chief_position, chief_velocity = classical_to_cartesian(
        a_m=7_200_000.0,
        e=0.02,
        inc_deg=48.0,
        raan_deg=25.0,
        argp_deg=35.0,
        true_anomaly_deg=15.0,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    deputy_position, deputy_velocity = classical_to_cartesian(
        a_m=7_205_000.0,
        e=0.021,
        inc_deg=48.1,
        raan_deg=25.2,
        argp_deg=34.8,
        true_anomaly_deg=15.3,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    chief = state(chief_position, chief_velocity)
    deputy = state(deputy_position, deputy_velocity)
    relative = absolute_to_classical_relative_orbital_elements(
        chief,
        deputy,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    assert isinstance(relative, ClassicalRelativeOrbitalElements)
    recovered = classical_relative_orbital_elements_to_absolute_state(
        chief,
        relative,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    np.testing.assert_allclose(recovered.r_m, deputy.r_m, atol=1e-6)
    np.testing.assert_allclose(recovered.v_mps, deputy.v_mps, atol=1e-9)


def test_relative_element_propagation_advances_only_relative_longitude() -> None:
    radius_m = 7_000_000.0
    chief = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )
    initial = RelativeOrbitalElements(
        delta_a=2.0e-4,
        delta_lambda_rad=-0.004,
        delta_ex=1.0e-4,
        delta_ey=-2.0e-4,
        delta_ix_rad=3.0e-4,
        delta_iy_rad=-4.0e-4,
    )
    history = propagate_relative_orbital_elements(
        initial,
        [0.0, 1_800.0],
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    deputy_a_m = radius_m * (1.0 + initial.delta_a)
    expected_rate = np.sqrt(EARTH.mu_m3ps2 / deputy_a_m**3) - np.sqrt(
        EARTH.mu_m3ps2 / radius_m**3
    )
    expected_constants = np.repeat(
        initial.as_vector()[None, [0, 2, 3, 4, 5]],
        history.shape[0],
        axis=0,
    )
    np.testing.assert_allclose(history[:, [0, 2, 3, 4, 5]], expected_constants)
    assert history[-1, 1] == pytest.approx(
        initial.delta_lambda_rad + expected_rate * 1_800.0
    )


def test_classical_relative_element_propagation_advances_mean_anomaly() -> None:
    radius_m = 7_000_000.0
    chief = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )
    initial = ClassicalRelativeOrbitalElements(
        delta_a_m=1_500.0,
        delta_e=1.0e-4,
        delta_i_rad=2.0e-4,
        delta_raan_rad=-3.0e-4,
        delta_argp_rad=4.0e-4,
        delta_mean_anomaly_rad=-0.002,
    )
    history = propagate_relative_orbital_elements(
        initial,
        [0.0, 900.0],
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
        representation="classical_elements",
    )
    expected_rate = np.sqrt(
        EARTH.mu_m3ps2 / (radius_m + initial.delta_a_m) ** 3
    ) - np.sqrt(EARTH.mu_m3ps2 / radius_m**3)

    assert history[-1, 5] == pytest.approx(
        initial.delta_mean_anomaly_rad + expected_rate * 900.0
    )
    np.testing.assert_allclose(
        history[:, [0, 1, 2, 3, 4]],
        np.repeat(
            initial.as_vector()[None, [0, 1, 2, 3, 4]],
            2,
            axis=0,
        ),
    )


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

    assert (
        np.linalg.norm(perturbed.relative_states_ric[-1] - unperturbed.relative_states_ric[-1])
        > 1e-4
    )


def test_relative_element_propagation_accepts_j2_force_model() -> None:
    chief_position, chief_velocity = classical_to_cartesian(
        a_m=7_000_000.0,
        e=0.001,
        inc_deg=40.0,
        raan_deg=20.0,
        argp_deg=10.0,
        true_anomaly_deg=30.0,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    chief = state(chief_position, chief_velocity)
    initial = RelativeOrbitalElements(
        delta_a=2.0e-4,
        delta_lambda_rad=-0.004,
        delta_ex=1.0e-4,
        delta_ey=-2.0e-4,
        delta_ix_rad=3.0e-4,
        delta_iy_rad=-4.0e-4,
    )
    times = np.asarray([0.0, 1_800.0, 3_600.0])
    unperturbed = propagate_relative_orbital_elements(
        initial,
        times,
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    perturbed = propagate_relative_orbital_elements(
        initial,
        times,
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
        perturbations=Perturbations(j2=True),
        max_step_s=5.0,
    )
    ric_history = propagate_relative_elements_to_ric(
        initial,
        times,
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
        perturbations=Perturbations(j2=True),
        max_step_s=5.0,
    )

    np.testing.assert_allclose(perturbed[0, 0:6], initial.as_vector(), atol=2.0e-14)
    assert np.linalg.norm(perturbed[-1, 0:6] - unperturbed[-1, 0:6]) > 1.0e-5
    np.testing.assert_allclose(ric_history[:, 6], times)
    assert np.all(np.isfinite(ric_history))


def test_relative_element_perturbations_require_zero_time_endpoint() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    chief = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )
    initial = RelativeOrbitalElements(
        delta_a=0.0,
        delta_lambda_rad=-0.001,
        delta_ex=1.0e-4,
        delta_ey=0.0,
        delta_ix_rad=0.0,
        delta_iy_rad=0.0,
    )

    with pytest.raises(ValueError, match="first or last"):
        propagate_relative_elements_to_ric(
            initial,
            [10.0, 20.0],
            chief_initial_state_eci=chief,
            mu_m3ps2=EARTH.mu_m3ps2,
            perturbations=Perturbations(j2=True),
        )


def test_relative_element_perturbations_propagate_backward_to_initial_epoch() -> None:
    chief_position, chief_velocity = classical_to_cartesian(
        a_m=7_000_000.0,
        e=0.001,
        inc_deg=40.0,
        raan_deg=20.0,
        argp_deg=10.0,
        true_anomaly_deg=30.0,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    chief = state(chief_position, chief_velocity)
    initial = RelativeOrbitalElements(
        delta_a=2.0e-4,
        delta_lambda_rad=-0.002,
        delta_ex=1.0e-4,
        delta_ey=-2.0e-4,
        delta_ix_rad=3.0e-4,
        delta_iy_rad=-4.0e-4,
    )
    times = np.asarray([-600.0, -300.0, 0.0])

    history = propagate_relative_orbital_elements(
        initial,
        times,
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
        perturbations=Perturbations(j2=True),
        max_step_s=5.0,
    )

    np.testing.assert_allclose(history[:, 6], times)
    np.testing.assert_allclose(history[-1, 0:6], initial.as_vector(), atol=2.0e-14)
    assert np.all(np.isfinite(history))


def test_coupled_propagation_accepts_absolute_deputy_initial_state() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    chief = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )
    deputy = state(
        [radius_m + 100.0, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / (radius_m + 100.0)), 0.0],
    )

    result = propagate_relative_numerical(
        chief,
        None,
        [0.0, 10.0],
        deputy_initial_eci=deputy,
        perturbations=Perturbations(j2=True),
    )

    np.testing.assert_allclose(result.deputy_states_eci[0, 0:3], deputy.r_m)
    np.testing.assert_allclose(result.deputy_states_eci[0, 3:6], deputy.v_mps)


def test_exact_nonlinear_ric_matches_coupled_two_body_propagation() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    chief = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )
    initial = state([500.0, -800.0, 100.0], [0.05, -0.02, 0.01])
    times = np.linspace(0.0, 600.0, 7)
    coupled = propagate_relative_numerical(
        chief,
        initial,
        times,
        max_step_s=0.5,
    )
    direct = propagate_nonlinear_relative_ric(
        np.hstack([initial.r_m, initial.v_mps]),
        times,
        mu_m3ps2=EARTH.mu_m3ps2,
        chief_orbit_radius_m=radius_m,
        max_step_s=0.5,
    )
    np.testing.assert_allclose(
        direct[:, 0:6],
        coupled.relative_states_ric,
        atol=2e-6,
    )


def test_coupled_ric_derivative_matches_eccentric_absolute_propagation() -> None:
    chief_position, chief_velocity = classical_to_cartesian(
        a_m=7_500_000.0,
        e=0.1,
        inc_deg=35.0,
        raan_deg=20.0,
        argp_deg=15.0,
        true_anomaly_deg=40.0,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    chief = state(chief_position, chief_velocity)
    relative = state([500.0, -800.0, 100.0], [0.05, -0.02, 0.01])
    derivative = coupled_relative_ric_derivative(
        np.hstack(
            [
                chief.r_m,
                chief.v_mps,
                relative.r_m,
                relative.v_mps,
            ]
        ),
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    interval_s = 0.01
    propagated = propagate_relative_numerical(
        chief,
        relative,
        [0.0, interval_s],
        max_step_s=1.0e-4,
    )
    finite_difference = (
        propagated.relative_states_ric[1]
        - propagated.relative_states_ric[0]
    ) / interval_s

    np.testing.assert_allclose(
        derivative[6:9],
        finite_difference[0:3],
        atol=1.1e-5,
    )
    np.testing.assert_allclose(
        derivative[9:12],
        finite_difference[3:6],
        atol=3e-8,
    )


def test_exact_nonlinear_ric_linearizes_to_cwh_near_the_chief() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    mean_motion = np.sqrt(EARTH.mu_m3ps2 / radius_m**3)
    relative_state = np.asarray([1.0, -2.0, 0.5, 1e-3, -2e-3, 5e-4])
    exact = nonlinear_relative_ric_derivative(
        relative_state,
        mu_m3ps2=EARTH.mu_m3ps2,
        chief_orbit_radius_m=radius_m,
    )
    linear = cwh_derivative(relative_state, mean_motion)
    np.testing.assert_allclose(exact, linear, rtol=0.0, atol=2e-12)


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
