from __future__ import annotations

import numpy as np
import pytest

from octavian import EARTH, Dynamics, state
from octavian.coordinates import RELATIVE_CARTESIAN, lvlh
from octavian.relative import (
    ClohessyWiltshire,
    cwh_derivative,
    cwh_rendezvous_velocity,
    cwh_state_transition,
    inertial_to_relative_state,
    propagate_cwh,
    relative_to_inertial_state,
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


def test_inertial_relative_state_transform_round_trip() -> None:
    chief = state([7_000_000.0, 0.0, 0.0], [0.0, 7_500.0, 0.0])
    relative = state([100.0, -250.0, 40.0], [0.2, -0.1, 0.05])
    deputy = relative_to_inertial_state(chief, relative)
    recovered = inertial_to_relative_state(chief, deputy)

    assert recovered.r_m == pytest.approx(relative.r_m, abs=1e-10)
    assert recovered.v_mps == pytest.approx(relative.v_mps, abs=1e-12)


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
